import json
import random
import pandas as pd
import numpy as np
import dataframe_image as dfi

from bs4 import BeautifulSoup
import sys

sys.path.append('../models/')
from utils import read_json, read_jsonl, write_json, write_jsonl, parquet2list


def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')


def highlight_rows(x, color):
    colors = color
    return ['background-color: {}'.format(colors[i % len(colors)]) for i in range(len(x))]


def render_mpl_table(data, savename):
    color = random.choice([
        ['#e6f7ff', '#ffffff'],
        ['#ffffff', '#d9f2e6'],
        ['#ffffff', '#f3e6ff'],
        ['#ffe6f2', '#ffffff'],
        ['#fff9e6', '#ffffff'],
        ['#ffffff', '#ffe6cc'],
        ['#ffffff', '#e6f2ff'],
        ['#e6fff2', '#ffffff'],
        ['#fff5e6', '#ffffff'],
        ['#ffffff', '#ffe6e6'],
        ['#ffffff', '#f2f2f2'],

    ])
    df_styled = data.style.apply(highlight_rows, axis=0, args=(color,))

    dfi.export(df_styled, savename, table_conversion='matplotlib')


def html_table_to_dataframe(html_table: str) -> pd.DataFrame:
    soup = BeautifulSoup(html_table, "html.parser")

    headers = []
    rows = []

    for row in soup.find_all('tr'):
        cols = row.find_all('td')
        row_data = []

        for col in cols:
            colspan = int(col.get('colspan', 1))
            cell_text = col.get_text(strip=True)
            row_data.extend([cell_text] * colspan)
        if not headers:
            headers = row_data
        else:
            rows.append(row_data)

    max_columns = max(len(headers), max([len(row) for row in rows], default=0))

    headers += [''] * (max_columns - len(headers))
    for row in rows:
        row += [''] * (max_columns - len(row))

    df = pd.DataFrame(rows, columns=headers)

    return df


def parse_html_tables(html_list):
    dataframes = []
    for html in html_list:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find('table')

        headers = []
        rows = []

        # Get table headers
        for th in table.find_all('th'):
            headers.append(th.get_text(strip=True))

        # In case headers are in the first row instead of <th> tags
        if not headers:
            headers = [td.get_text(strip=True) for td in table.find_all('tr')[0].find_all('td')]

        # Get all the rows of the table
        for tr in table.find_all('tr')[1:]:  # Skip header row
            row = [td.get_text(strip=True) for td in tr.find_all('td')]
            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        dataframes.append(df)

    return dataframes


with open('dev.json', 'r') as f:
    data = json.load(f)
import os
from tqdm import tqdm

formated_all_data = []
os.makedirs('images', exist_ok=True)
for sample in tqdm(data):
    uid = sample['uid']
    tables = sample['tables']
    images_path = []
    for tid, table in enumerate(tables):

        if os.path.exists(f"images/{uid}_{tid}.png"):
            images_path.append('../multihiertt/' + f"images/{uid}_{tid}.png")

        df = html_table_to_dataframe(table)


        def make_unique_columns(df):
            cols = pd.Series(df.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols[cols == dup].index.values.tolist()] = [f'{dup}_{i}' if i != 0 else dup for i in
                                                                 range(sum(cols == dup))]
            df.columns = cols
            return df


        df = make_unique_columns(df)

        render_mpl_table(
            df,
            savename=f"images/{uid}_{tid}.png",
            # col_width=4.0,
        )
    if len(images_path) == len(tables):
        question = sample['qa']['question']
        answer = sample['qa']['answer']
        formated_one_sample = {"images_path": images_path,
                               'question': question,
                               "answers": answer,
                               'ques_type': 'open-ended'}

        formated_all_data.append(formated_one_sample)
print(f"prepared {len(formated_all_data)} samples for MultiHiertt dataset.")
write_jsonl('../eval_multihiertt.jsonl', formated_all_data)
