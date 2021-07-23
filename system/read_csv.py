import pandas as pd

df = pd.read_csv("test_data.csv")


def retrieve_id(id: int):
    row = df.loc[df.ID == id].iloc[0]
    content = row["content"][:400] + "..."
    date = row["publish_date"]
    if type(date) == str:
        date = date.split()[0]
    else:
        date = "Not available"
    return {
        "id": row["ID"],
        "source": row["source"],
        "title": row["title"],
        "content": content,
        "date": date,
        "link": row["link"],
    }
