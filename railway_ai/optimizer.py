"""MILP-based scheduler for railway dispatch.

This module formulates the real-time dispatching problem as a mixed
integer linear program using the ``pulp`` library. The solver
determines arrival and departure times at each station for each train
subject to run-time, dwell and headway constraints. The objective is
configurable via the supplied rules but defaults to minimising weighted
departure times to encourage overall punctuality.

The formulation used here is intentionally simplified for clarity:

* All sections are treated as single-track resources, meaning at most
  one train may occupy a section at a time. The ``tracks`` field in
  ``sections.csv`` is currently ignored; extension to multi-track
  resources is possible by relaxing the headway constraints or
  introducing multiple precedence variables per track.
* We generate precedence binary variables for each pair of trains on
  each section. These variables decide which train traverses the
  section first. Big-M constraints enforce non-overlap between trains.
* The objective minimises the sum of departure times at all stations
  weighted by the train priority. More sophisticated objectives (e.g.
  priority-weighted delay w.r.t. timetables) can be implemented by
  adjusting the coefficients.

Even with these simplifications, the model can demonstrate realistic
reordering and retiming decisions under disruption. You can add new
trains, specify current delays or block sections by passing
appropriate parameters to ``optimise_schedule``.
"""

from __future__ import annotations

import itertools
import json
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pulp


def _build_routes(trains: pd.DataFrame, stations: pd.DataFrame) -> Dict[int, List[int]]:
    """Return a mapping from train_id to ordered list of station_ids.

    A train's direction determines whether the sequence follows the
    station list forward (direction=1) or backwards (direction=-1).
    """
    station_ids = list(stations["id"])
    routes: Dict[int, List[int]] = {}
    for _, row in trains.iterrows():
        if row["direction"] == 1:
            routes[row["id"]] = station_ids
        else:
            routes[row["id"]] = list(reversed(station_ids))
    return routes


def _build_section_lookup(sections: pd.DataFrame) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Create a lookup mapping between station pairs and section parameters.

    Because the input sections are directional agnostic, we record each
    connection in both orders (from→to and to→from). Each entry maps
    to a dict with keys ``runtime`` (base run time in minutes) and
    ``tracks`` (number of parallel tracks). The tracks field is not
    currently used but included for completeness.
    """
    lookup: Dict[Tuple[int, int], Dict[str, float]] = {}
    for _, sec in sections.iterrows():
        lookup[(sec["from_station_id"], sec["to_station_id"])] = {
            "runtime": float(sec["base_runtime_min"]),
            "tracks": int(sec["tracks"]),
        }
        lookup[(sec["to_station_id"], sec["from_station_id"])] = {
            "runtime": float(sec["base_runtime_min"]),
            "tracks": int(sec["tracks"]),
        }
    return lookup


def optimise_schedule(
    trains_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    sections_df: pd.DataFrame,
    rules: Dict,
    current_delays: Optional[Dict[int, float]] = None,
    new_train: Optional[Dict] = None,
    time_horizon_min: Optional[float] = None,
) -> pd.DataFrame:
    """Solve for an updated schedule given the current network state.

    :param trains_df: DataFrame of trains with columns ``id``, ``train_type``,
        ``priority``, ``direction`` and ``earliest_departure_min``.
    :param stations_df: DataFrame of stations with at least the column ``id``.
    :param sections_df: DataFrame of sections with columns ``from_station_id``,
        ``to_station_id``, ``base_runtime_min`` and ``tracks``.
    :param rules: dictionary of operational rules containing ``headway_minutes``,
        ``dwell_minutes`` and ``speed_factors``.
    :param current_delays: optional mapping of train_id to additional minutes
        of delay to apply to their earliest departure times.
    :param new_train: optional dict describing a new train to be inserted.
        Should include the same fields as ``trains.csv``.
    :param time_horizon_min: optional horizon cap; arrival/departure times
        beyond this will be penalised heavily in the objective (not used here).
    :return: DataFrame of optimised arrival and departure times for each
        train and station (columns: train_id, station_id, arrival_min, departure_min).
    """
    # If a new train is provided, append it to the trains dataframe
    trains_df = trains_df.copy()
    if new_train:
        # assign new id if not given
        new_id = int(new_train.get("id", trains_df["id"].max() + 1))
        new_record = {
            "id": new_id,
            "name": new_train.get("name", f"NEW{new_id}"),
            "train_type": new_train.get("train_type", "PASSENGER"),
            "priority": new_train.get("priority", 1),
            "direction": new_train.get("direction", 1),
            "earliest_departure_min": new_train.get("earliest_departure_min", 0),
        }
        trains_df = pd.concat([trains_df, pd.DataFrame([new_record])], ignore_index=True)

    # Apply current delays: shift earliest departure times
    if current_delays:
        for t_id, delay in current_delays.items():
            idx = trains_df.index[trains_df["id"] == t_id].tolist()
            if idx:
                i = idx[0]
                trains_df.at[i, "earliest_departure_min"] += delay

    # Build structures
    routes = _build_routes(trains_df, stations_df)
    section_lookup = _build_section_lookup(sections_df)
    headway = rules.get("headway_minutes", 5.0)
    dwell_map = rules.get("dwell_minutes", {})
    speed_map = rules.get("speed_factors", {})
    priority_map = trains_df.set_index("id")["priority"].to_dict()
    direction_map = trains_df.set_index("id")["direction"].to_dict()
    earliest_departure_map = (
        trains_df.set_index("id")["earliest_departure_min"].astype(float).to_dict()
    )
    train_type_map = trains_df.set_index("id")["train_type"].to_dict()

    # Decision variables
    problem = pulp.LpProblem("RailwayDispatch", pulp.LpMinimize)
    # a[t, i] and d[t, i] arrival and departure times
    a_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
    d_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
    # Create variables per train and station along its route
    for t_id, station_list in routes.items():
        for s_id in station_list:
            a_vars[(t_id, s_id)] = pulp.LpVariable(f"a_{t_id}_{s_id}", lowBound=0)
            d_vars[(t_id, s_id)] = pulp.LpVariable(f"d_{t_id}_{s_id}", lowBound=0)

    # Objective: minimise weighted sum of departure times at all stations
    # This encourages early departures and thus throughput
    objective = pulp.lpSum(
        priority_map.get(t_id, 1) * d_vars[(t_id, s_id)]
        for t_id, station_list in routes.items()
        for s_id in station_list
    )
    problem += objective

    # Constraints
    # 1. Dwell: d >= a + dwell
    for t_id, station_list in routes.items():
        train_type = trains_df.loc[trains_df["id"] == t_id, "train_type"].iloc[0]
        dwell = dwell_map.get(train_type, 5.0)
        for s_id in station_list:
            problem += d_vars[(t_id, s_id)] >= a_vars[(t_id, s_id)] + dwell
    # 2. Run time between consecutive stations
    for t_id, station_list in routes.items():
        train_type = trains_df.loc[trains_df["id"] == t_id, "train_type"].iloc[0]
        speed_factor = speed_map.get(train_type, 1.0)
        for i in range(len(station_list) - 1):
            s_from = station_list[i]
            s_to = station_list[i + 1]
            runtime = section_lookup[(s_from, s_to)]["runtime"] / speed_factor
            problem += a_vars[(t_id, s_to)] >= d_vars[(t_id, s_from)] + runtime
    # 3. Earliest departure at origin
    for t_id, station_list in routes.items():
        origin = station_list[0]
        earliest = float(trains_df.loc[trains_df["id"] == t_id, "earliest_departure_min"].iloc[0])
        problem += a_vars[(t_id, origin)] >= earliest
        problem += d_vars[(t_id, origin)] >= earliest
    # 4. Headway and precedence constraints: ensure no overlap on sections
    # For each section and pair of trains using that section, add precedence variables and constraints
    M = 10_000  # big-M constant
    precedence_vars: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
    # Build list of train pairs and section indices
    for s in sections_df.itertuples(index=False):
        sec_from = s.from_station_id
        sec_to = s.to_station_id

        def section_entry_index(route: List[int], from_id: int, to_id: int) -> Optional[int]:
            for idx in range(len(route) - 1):
                if (route[idx] == from_id and route[idx + 1] == to_id) or (
                    route[idx] == to_id and route[idx + 1] == from_id
                ):
                    return idx
            return None

        # Gather trains that traverse this section along with entry indices
        trains_on_section: List[int] = []
        entry_indices: Dict[int, int] = {}
        for t_id, st_list in routes.items():
            idx = section_entry_index(st_list, sec_from, sec_to)
            if idx is None:
                continue
            trains_on_section.append(t_id)
            entry_indices[t_id] = idx

        # Enforce headway for trains moving in the same direction using deterministic ordering
        for direction_value in {1, -1}:
            same_direction = [
                t_id for t_id in trains_on_section if direction_map.get(t_id, 1) == direction_value
            ]
            if len(same_direction) <= 1:
                continue
            same_direction.sort(key=lambda tid: earliest_departure_map.get(tid, 0.0))
            for prev_id, next_id in zip(same_direction, same_direction[1:]):
                idx_prev = entry_indices[prev_id]
                idx_next = entry_indices[next_id]
                route_prev = routes[prev_id]
                route_next = routes[next_id]
                entry_prev = route_prev[idx_prev]
                entry_next = route_next[idx_next]
                exit_prev = route_prev[idx_prev + 1]
                train_type_prev = train_type_map.get(prev_id)
                runtime_prev = section_lookup[(entry_prev, exit_prev)]["runtime"] / speed_map.get(
                    train_type_prev, 1.0
                )
                problem += (
                    d_vars[(next_id, entry_next)]
                    >= d_vars[(prev_id, entry_prev)] + runtime_prev + headway
                )

        # Skip binary precedence if section has multiple tracks (sufficient capacity)
        if getattr(s, "tracks", 1) > 1:
            continue

        # For every unordered pair of trains travelling in opposite directions, impose ordering
        for t1, t2 in itertools.combinations(trains_on_section, 2):
            if direction_map.get(t1, 1) == direction_map.get(t2, 1):
                continue
            route1 = routes[t1]
            route2 = routes[t2]
            idx1 = entry_indices[t1]
            idx2 = entry_indices[t2]
            entry1 = route1[idx1]
            entry2 = route2[idx2]
            exit1 = route1[idx1 + 1]
            exit2 = route2[idx2 + 1]
            train_type1 = train_type_map.get(t1)
            train_type2 = train_type_map.get(t2)
            runtime1 = section_lookup[(entry1, exit1)]["runtime"] / speed_map.get(train_type1, 1.0)
            runtime2 = section_lookup[(entry2, exit2)]["runtime"] / speed_map.get(train_type2, 1.0)
            prec_var = pulp.LpVariable(f"y_{t1}_{t2}_{s.id}", cat="Binary")
            precedence_vars[(t1, t2, s.id)] = prec_var
            problem += (
                d_vars[(t2, entry2)]
                >= d_vars[(t1, entry1)] + runtime1 + headway - M * (1 - prec_var)
            )
            problem += (
                d_vars[(t1, entry1)]
                >= d_vars[(t2, entry2)] + runtime2 + headway - M * (prec_var)
            )

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)
    # Extract solution
    results = []
    for t_id, station_list in routes.items():
        for s_id in station_list:
            arrival = pulp.value(a_vars[(t_id, s_id)])
            departure = pulp.value(d_vars[(t_id, s_id)])
            results.append(
                {
                    "train_id": t_id,
                    "station_id": s_id,
                    "arrival_min": arrival,
                    "departure_min": departure,
                }
            )
    return pd.DataFrame(results)


def plot_schedule(
    schedule_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    *,
    show: bool = True,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Create a time–space diagram of the optimised dispatch plan.

    The horizontal axis corresponds to minutes since the start of the horizon,
    and the vertical axis lists stations in order. Each train is plotted as a
    coloured polyline linking its arrival and departure times at each station.
    """

    if schedule_df.empty:
        raise ValueError("schedule_df must not be empty when plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    station_order = stations_df.sort_values("id").reset_index(drop=True)
    station_positions = {
        station_id: idx for idx, station_id in enumerate(station_order["id"].tolist())
    }
    station_labels = (
        station_order["name"] if "name" in station_order else station_order["id"]
    )

    for train_id, group in schedule_df.groupby("train_id", sort=False):
        group = group.copy()
        group["station_pos"] = group["station_id"].map(station_positions)
        group.sort_values("station_pos", inplace=True)

        times: List[float] = []
        positions: List[int] = []
        for _, row in group.iterrows():
            y_pos = row["station_pos"]
            times.extend([row["arrival_min"], row["departure_min"]])
            positions.extend([y_pos, y_pos])

        ax.plot(times, positions, marker="o", linewidth=2, label=f"Train {train_id}")

    ax.set_xlabel("Time [minutes]")
    ax.set_ylabel("Station")
    ax.set_yticks(range(len(station_order)))
    ax.set_yticklabels(station_labels)
    ax.set_title("Optimised Railway Schedule")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimise train schedule using MILP")
    parser.add_argument("--data-dir", type=str, default="./synthetic_data", help="Path to input data directory")
    parser.add_argument("--rules", type=str, default="rules.json", help="Filename of rules JSON in data directory")
    parser.add_argument("--new-train", type=str, help="JSON string describing a new train to insert")
    parser.add_argument("--current-delays", type=str, help="JSON string mapping train_id to delay minutes")
    parser.add_argument("--output", type=str, help="Optional CSV file to write results to")
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display a time–space diagram of the optimised schedule",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        help="Optional filename to save the time–space diagram image",
    )
    args = parser.parse_args()
    from pathlib import Path
    data_dir = Path(args.data_dir)
    trains_df = pd.read_csv(data_dir / "trains.csv")
    stations_df = pd.read_csv(data_dir / "stations.csv")
    sections_df = pd.read_csv(data_dir / "sections.csv")
    with open(data_dir / args.rules, "r", encoding="utf-8") as f:
        rules = json.load(f)
    new_train = json.loads(args.new_train) if args.new_train else None
    current_delays = json.loads(args.current_delays) if args.current_delays else None
    schedule = optimise_schedule(
        trains_df, stations_df, sections_df, rules, current_delays, new_train
    )
    if args.output:
        schedule.to_csv(args.output, index=False)
        print(f"Optimised schedule written to {args.output}")
    else:
        print(schedule)

    if args.show_plot or args.plot_output:
        plot_schedule(
            schedule,
            stations_df,
            show=args.show_plot,
            save_path=args.plot_output,
        )
