import glob

import pandas as pd

from name_dict import NAME_DICT

BB_SIZE = 4


all_files = glob.glob("poker_log/*.csv")

li = []

for f in all_files:
    df = pd.read_csv(f, index_col="order")
    li.append(df)

df = pd.concat(li, axis=0)

df = df.sort_index()
df = df.reset_index(drop=True)
df = df[["entry"]]

#%%
# Round No
df["round_no"] = df["entry"].str.contains("-- starting hand").cumsum()

#%%
# Pre-flop
df["is_small_blind"] = df["entry"].str.contains("posts a small blind of")
df["is_open_flop"] = df["entry"].str.contains("Flop:")
df.loc[df["is_small_blind"], "is_preflop"] = True
df.loc[df["is_open_flop"], "is_preflop"] = False
df["is_preflop"] = df["is_preflop"].fillna(method="pad").fillna(False)
df = df.drop(columns=["is_small_blind", "is_open_flop"])

#%%
# Player name
df["player_name"] = df["entry"].str.extract(r"\"(\S+) @ \S+\"", expand=False)
df["player_name"] = df["player_name"].replace(NAME_DICT)

assert set(df["player_name"].dropna()).issubset(
    set(NAME_DICT.values())
), "Please add more name dict"

#%%
# Stack
stack = df.set_index("round_no")
is_stack = stack["entry"].str.contains("Player stacks:")
stack = stack.loc[is_stack, "entry"].str.extractall(
    r"\"(?P<player_name>\S+) @ \S+\" \((?P<stack>\d+)\)"
)

stack["stack"] = pd.to_numeric(stack["stack"])
stack["stack"] = stack["stack"] / BB_SIZE

stack["player_name"] = stack["player_name"].replace(NAME_DICT)

stack = stack.reset_index("round_no")
stack = stack.reset_index(drop=True)

df = df.merge(stack, on=["player_name", "round_no"], how="left", validate="m:1")

# Drop less than 3 player round
player_no = stack.groupby("round_no").count()
drop_round = set(player_no[player_no["player_name"] < 3].index)
df = df[~df["round_no"].isin(drop_round)]

#%%
# Position
position = df.copy()
position["position"] = df["entry"].str.extract(
    r"(small blind|big blind|dealer)", expand=False
)

position = position[["player_name", "round_no", "position"]].dropna()
position = position.drop_duplicates(
    ["round_no", "position"], keep="first"
)  # Sit while playing would pay SB and BB

df = df.merge(position, on=["player_name", "round_no"], how="left", validate="m:1")
is_name = df["player_name"].notna()
df.loc[is_name] = df.loc[is_name].fillna({"position": "middle position"})

#%%
# Action
df["action"] = df["entry"].str.extract(
    r"(calls \d+|bets \d+|raises to \d+|checks|folds)"
)
df["sizing"] = pd.to_numeric(df["action"].str.extract(r"(\d+)", expand=False))
df["sizing"] = df["sizing"] / BB_SIZE

df["action"] = df["action"].str.extract(r"(call|bet|raise|check|fold)")

is_action = df["action"].notna()
df.loc[is_action] = df.loc[is_action].fillna({"sizing": 0})

#%%
# Hand
hand = df.copy()
hand["hand"] = (
    hand["entry"]
    .str.extract(r"(shows a .*)", expand=False)
    .str.split("shows a ")
    .str[-1]
    .str[:-1]
)
hand = hand[["round_no", "player_name", "hand"]].dropna()

hand[["hand1", "hand2"]] = hand["hand"].str.split(",", expand=True)
hand["hand1"] = hand["hand1"].str.strip()
hand["hand2"] = hand["hand2"].str.strip()

hand["hand1_rank"] = hand["hand1"].str[:-1]
hand["hand1_suit"] = hand["hand1"].str[-1]
hand["hand2_rank"] = hand["hand2"].str[:-1]
hand["hand2_suit"] = hand["hand2"].str[-1]

hand[["hand1_rank", "hand2_rank"]] = hand[["hand1_rank", "hand2_rank"]].replace(
    {"A": "14", "J": "11", "Q": "12", "K": "13"}
)
hand["hand1_rank"] = pd.to_numeric(hand["hand1_rank"])
hand["hand2_rank"] = pd.to_numeric(hand["hand2_rank"])

hand[["hand1_suit", "hand2_suit"]] = hand[["hand1_suit", "hand2_suit"]].replace(
    {"♠": "spade", "♥": "heart", "♦": "diamond", "♣": "club"}
)
hand = hand[
    ["round_no", "player_name", "hand1_rank", "hand1_suit", "hand2_rank", "hand2_suit"]
]

hand = hand.drop_duplicates()

df = df.merge(hand, on=["player_name", "round_no"], how="left", validate="m:1")

#%%
# Export for NN
out = df.loc[
    df["is_preflop"],
    [
        "player_name",
        "stack",
        "position",
        "action",
        "sizing",
        "hand1_rank",
        "hand1_suit",
        "hand2_rank",
        "hand2_suit",
    ],
]
out = out.dropna().reset_index(drop=True)

out["is_connect"] = (out["hand1_rank"] - out["hand2_rank"]).abs().isin({1, 12})
out["is_suit"] = out["hand1_suit"] == out["hand2_suit"]
out["is_premium"] = (out["hand1_rank"] >= 10) & (out["hand2_rank"] >= 10)
out["is_pocket"] = out["hand1_rank"] == out["hand2_rank"]

out = out[
    [
        "player_name",
        "stack",
        "position",
        "action",
        "sizing",
        "is_connect",
        "is_suit",
        "is_premium",
        "is_pocket",
    ]
]

out.to_csv("pk_pre_flop_clean.csv")
