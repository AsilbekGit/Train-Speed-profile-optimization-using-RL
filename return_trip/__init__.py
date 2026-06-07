"""
return_trip — train the same algorithms on the *return* leg (B → A).

The forward route stored in data/data.csv goes from station A to station B.
For the return trip, the train traverses the same physical track in reverse:

  - Segment order is reversed.
  - Grade SIGNS flip: a forward +5% uphill becomes a -5% downhill on return.
  - Speed limits and curvatures are direction-agnostic — same values, just
    reversed in segment order.

Run from the project root:
    python -m return_trip.train_qsarsa
    python -m return_trip.train_dqn   --steps 1500000
    python -m return_trip.train_ppo   --steps 1500000

The algorithm code itself is unchanged — we just feed reversed data through
the existing TrainEnv (which uses the Davis equation) and the same trainers
used for the forward leg. The agent does NOT need a "direction" flag; the
return trip is just a different route from its perspective.
"""
