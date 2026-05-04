import pandas as pd
import sys
import json
from typing import List
import numpy as np
from openai import OpenAI
import os
def find_nearest_neighbors_by_entity_rank(
    entity_name: str,
    all_entities: dict,
    all_relationships: dict,
    exclude_entity_names: list[str] | None = None,
    k: int | None = 10,
) -> list:
    """Retrieve entities that have direct connections with the target entity, sorted by entity rank."""
    if exclude_entity_names is None:
        exclude_entity_names = []
    entity_relationships = [
        rel
        for rel in all_relationships.values()
        if rel["feature_1"] == entity_name or rel["feature_2"] == entity_name
    ]
    # for rel in entity_relationships:
    #     print(rel)
    source_entity_names = {rel["feature_1"] for rel in entity_relationships}
    target_entity_names = {rel["feature_2"] for rel in entity_relationships}
    related_entity_names = (source_entity_names.union(target_entity_names)).difference(
        set(exclude_entity_names)
    )
    return related_entity_names, entity_relationships


def get_data(
        local_dir: str,
        file_name: str,
        start_date: str ="",
        end_date: str = "",
        usecols: List[str] = None,
        date_column: str = "date",
    ) -> pd.DataFrame:
        if file_name == "":
            return pd.DataFrame()
        # local_dir = os.path.join(os.getcwd(), local_dir)
        # print(local_dir)
        if usecols is None:
            try:
                df = pd.read_csv(os.path.join(local_dir, file_name))
                # print(df)
            except FileNotFoundError:
                return pd.DataFrame()
        else:
            try:
                if date_column not in usecols:
                    usecols =[date_column] + usecols
                df = pd.read_csv(
                    os.path.join(local_dir, file_name),
                    usecols=usecols,
                )
                df = df[usecols]
            except FileNotFoundError:
                return pd.DataFrame(columns=usecols)
        # print(df)

        # df[date_column] = pd.to_datetime(df[date_column], errors='coerce',unit='ms')
        # print(df)
        if date_column == "date":
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        elif date_column == "timestamps":
            df[date_column] = pd.to_datetime(df[date_column], unit='ms')
        # Convert the "date" column to a datetime object with the format "YYYY-MM-DD"
        # if date_column == "date":
        #     df[date_column] = pd.to_datetime(
        #         df[date_column], format="%Y-%m-%d"
        #     )
        # else:
        #     df[date_column] = pd.to_datetime(
        #         df[date_column], unit="ms"
        #     )

        if end_date or end_date == start_date:
            # Filter the DataFrame to get the rows for the input dates (multiple dates)
            selected_rows = df[
                (
                    df[date_column]
                    >= pd.to_datetime(start_date, format="%Y-%m-%d")
                )
                & (
                    df[date_column]
                    <= pd.to_datetime(end_date, format="%Y-%m-%d")
                     + pd.Timedelta(days=1)
                )
            ]
        else:
            # Filter the DataFrame to get the rows for the input date (single dates)
            selected_rows = df[
                (
                    df[date_column]
                    == pd.to_datetime(start_date, format="%Y-%m-%d")
                )
            ]

        # Check if the input date exists in the DataFrame
        if selected_rows.empty:
            print(
                f"No data found between the date {start_date} and {end_date}."
            )
        return selected_rows

def get_data2(
        local_dir: str,
        file_name: str,
        start_date: str ="",
        end_date: str = "",
        usecols: List[str] = None,
        date_column: str = "date",
    ) -> pd.DataFrame:
        if file_name == "":
            return pd.DataFrame()
        # local_dir = os.path.join(os.getcwd(), local_dir)
        # print(local_dir)
        if usecols is None:
            try:
                df = pd.read_csv(os.path.join(local_dir, file_name))
                print(df)
            except FileNotFoundError:
                return pd.DataFrame()
        else:
            try:
                if "date" not in usecols:
                    usecols =["date"] + usecols
                df = pd.read_csv(
                    os.path.join(local_dir, file_name),
                    usecols=usecols,
                )
                df = df[usecols]
            except FileNotFoundError:
                return pd.DataFrame(columns=usecols)
        # print(df)
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        # Convert the "date" column to a datetime object with the format "YYYY-MM-DD"
        # if date_column == "date":
        #     df[date_column] = pd.to_datetime(
        #         df[date_column], format="%Y-%m-%d"
        #     )
        # else:
        #     df[date_column] = pd.to_datetime(
        #         df[date_column], unit="ms"
        #     )

        if end_date or end_date == start_date:
            # Filter the DataFrame to get the rows for the input dates (multiple dates)
            selected_rows = df[
                (
                    df[date_column]
                    >= pd.to_datetime(start_date, format="%Y-%m-%d")
                )
                & (
                    df[date_column]
                    <= pd.to_datetime(end_date, format="%Y-%m-%d")
                    # + pd.Timedelta(days=1)
                )
            ]
        else:
            # Filter the DataFrame to get the rows for the input date (single dates)
            selected_rows = df[
                (
                    df[date_column]
                    == pd.to_datetime(start_date, format="%Y-%m-%d")
                )
            ]

        # Check if the input date exists in the DataFrame
        if selected_rows.empty:
            print(
                f"No data found between the date {start_date} and {end_date}."
            )
        return selected_rows

def build_feature_context(
        matched_entities,
        related_entities,
        start_date: str ="2020-07-23",
        end_date: str = "2020-07-27",
):
    data = {}
    data_root_path = 'par_5'
    context ='Matched Entities:\n'
    for entity in matched_entities:
        feat = entity['node']
        if feat.if_data_associated:
            if len(feat.data['date']) > 0:
                id = feat.data['date'][0][0]
                path =feat.data['date'][0][2]
                df = get_data(data_root_path,path, start_date, end_date,[id])
            else:
                id = feat.data['timestamp'][0][0]
                path =feat.data['timestamp'][0][2]
                df = get_data(data_root_path,path, start_date, end_date,[id],date_column='timestamps')
            mark_data = df.to_markdown(index=False)
            trace = {
            "x": df['timestamps'].tolist(),
            "y": df[id].tolist(),
            }
            data[feat.name] = trace
            context += (
                f"{feat.name}:\n"
                f"description: {feat.description}\n"
                f"range: {feat.range}\n"
                f"recommendation: {feat.recommendation}\n"
                f"{mark_data}\n"
            )
        else:
            context += (
                f"{feat.name}:\n"
                f"description: {feat.description}\n"
                f"range: {feat.range}\n"
                f"recommendation: {feat.recommendation}\n"
                f"data: No data\n"
            )
    context += 'related entities:\n'
    for feat in related_entities:
        entity, weight = feat
        print(entity.name,'<<<<<<<<<<<')
        # print(feat)
        if entity.if_data_associated:
            if len(entity.data['date']) > 0:
                # print(feat.data)
                id = entity.data['date'][0][0]
                # print(id)
                path =entity.data['date'][0][2]
                # print(path)
                df = get_data(data_root_path,path, start_date, end_date,[id])
                mark_data = df.to_markdown(index=False)
                # trace = df.to_dict('records')
                trace = {
                "x": df['date'].tolist(),
                "y": df[id].tolist(),
                }
                data[entity.name] = trace
            # print(feat["data"])
                context += (
                    f"{entity.name}:\n"
                    f"weight: {weight}\n"
                    f"description: {entity.description}\n"
                    f"range: {entity.range}\n"
                    f"recommendation: {entity.recommendation}\n"
                    f"{mark_data}\n"
                )
        else:
            context += (
                f"{entity.name}:\n"
                f"weight: {weight}\n"
                f"description: {entity.description}\n"
                f"range: {entity.range}\n"
                f"recommendation: {entity.recommendation}\n"
                f"data: No data\n"
            )
    # print(context)
    return context, data
def build_edge_context(
        feature_relationships
):
    context =''
    for rel in feature_relationships:
        context += f"relationship between {rel.entity_1_name} and {rel.entity_2_name}:\n {rel.description} \n"

    # print(context)
    return context


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-3-small"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# def search_docs(df, user_query, top_n=4, to_print=True):
#     embedding = get_embedding(
#         user_query,
#         model="text-embedding-3-small"
#     )
#     df["similarities"] = df.name_embed.apply(lambda x: cosine_similarity(x, embedding))

#     res = (
#         df.sort_values("similarities", ascending=False)
#         .head(top_n)
#     )
#     # if to_print:
#     #     display(res)
#     return res

def search_docs(nodes_dict, user_query, top_n=4, to_print=True):
    # Get embedding for the query
    query_embedding = get_embedding(
        user_query,
        model="text-embedding-3-small"
    )
    
    # Calculate similarities for all nodes
    similarities = {
        node_id: cosine_similarity(node.name_embedding, query_embedding)
        for node_id, node in nodes_dict.items()
    }
    
    # Sort nodes by similarity and get top_n results
    sorted_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_results = sorted_nodes[:top_n]
    
    # Create result dictionary with node_id and similarity score
    results = [
        {
            'node': nodes_dict[node_id],
            'similarity': sim_score
        }
        for node_id, sim_score in top_results
    ]
    
    return results