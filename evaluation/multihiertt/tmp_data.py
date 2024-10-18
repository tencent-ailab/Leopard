import pandas as pd

import sys

sys.path.append('/Users/jiamengzhao/Downloads/Paper_cheetah/tencent_code/multi_img_data/data_process_scripts')


def convert_to_dataframe(data_dict):
    """
    Convert nested dictionaries (with years and financial data) into a pandas DataFrame.

    Args:
        data_dict (dict): A dictionary containing structured table data.
                          Example structure:
                          {
                              "year": [2007, 2006, 2005],
                              "group_retirement_products": {
                                  "premiums": [446, 386, 351],
                                  "investment_income": [2280, 2279, 2233],
                                  "capital_gains_losses": [-451, -144, -67],
                                  ...
                              },
                              "individual_fixed_annuities": {
                                  "premiums": [96, 122, 97],
                                  ...
                              },
                              ...
                          }

    Returns:
        pd.DataFrame: DataFrame containing the structured table data.
    """

    # Flatten the dictionary to prepare for DataFrame creation
    flattened_data = {}

    # Iterate over each main category (e.g., group_retirement_products, individual_fixed_annuities)
    for category, sub_dict in data_dict.items():
        if isinstance(sub_dict, dict):  # If the value is a dictionary
            for sub_category, values in sub_dict.items():
                # Create new column name by combining category and sub-category
                column_name = f"{category}_{sub_category}".replace(" ", "_").lower()
                flattened_data[column_name] = values
        else:
            # For non-dict items (e.g., year), just add them directly
            flattened_data[category] = sub_dict

    # Convert the flattened dictionary into a pandas DataFrame
    df = pd.DataFrame(flattened_data)

    return df


from bs4 import BeautifulSoup


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


data =  [
            "<table><tr><td> </td><td colspan=\"2\"> Outstanding Balance December 31,</td><td colspan=\"2\"> Stated Interest Rate December 31,</td></tr><tr><td> </td><td> 2009</td><td> 2008</td><td>2009</td><td> 2008</td></tr><tr><td> </td><td colspan=\"4\"> (in millions) </td></tr><tr><td>Senior notes due 2010</td><td>$340</td><td>$800</td><td>5.4%</td><td>5.4%</td></tr><tr><td>Senior notes due 2015</td><td>700</td><td>700</td><td>5.7</td><td>5.7</td></tr><tr><td>Senior notes due 2019</td><td>300</td><td>\u2014</td><td>7.3</td><td>\u2014</td></tr><tr><td>Senior notes due 2039</td><td>200</td><td>\u2014</td><td>7.8</td><td>\u2014</td></tr><tr><td>Junior subordinated notes due 2066</td><td>322</td><td>457</td><td>7.5</td><td>7.5</td></tr><tr><td>Floating rate revolving credit borrowings due 2013</td><td>142</td><td>64</td><td>4.1</td><td>3.6</td></tr><tr><td>Floating rate revolving credit borrowings due 2014</td><td>198</td><td>\u2014</td><td>5.9</td><td>\u2014</td></tr><tr><td>Floating rate revolving credit borrowings due 2014</td><td>41</td><td>\u2014</td><td>2.5</td><td>\u2014</td></tr><tr><td>Municipal bond inverse floater certificates due 2021</td><td>6</td><td>6</td><td>0.3</td><td>2.2</td></tr><tr><td>Total</td><td>$2,249</td><td>$2,027</td><td></td><td></td></tr></table>",
            "<table><tr><td> </td><td colspan=\"3\"> Years Ended December 31,</td></tr><tr><td> </td><td> 2009</td><td> 2008</td><td> 2007</td></tr><tr><td> </td><td colspan=\"3\"> (in millions) </td></tr><tr><td>Stock options</td><td>$53</td><td>$40</td><td>$37</td></tr><tr><td>Restricted stock awards</td><td>59</td><td>57</td><td>52</td></tr><tr><td>Restricted stock units</td><td>70</td><td>51</td><td>54</td></tr><tr><td>Total</td><td>$182</td><td>$148</td><td>$143</td></tr></table>",
            "<table><tr><td> </td><td colspan=\"2\">Years Ended December 31,</td><td></td><td></td></tr><tr><td> </td><td>2008</td><td>2007</td><td colspan=\"2\">Change</td></tr><tr><td> </td><td colspan=\"4\">(in millions, except percentages)</td></tr><tr><td> Revenues</td><td></td><td></td><td></td><td></td></tr><tr><td>Management and financial advice fees</td><td>$2,899</td><td>$3,238</td><td>$-339</td><td>-10%</td></tr><tr><td>Distribution fees</td><td>1,565</td><td>1,762</td><td>-197</td><td>-11</td></tr><tr><td>Net investment income</td><td>817</td><td>2,014</td><td>-1,197</td><td>-59</td></tr><tr><td>Premiums</td><td>1,048</td><td>1,017</td><td>31</td><td>3</td></tr><tr><td>Other revenues</td><td>766</td><td>724</td><td>42</td><td>6</td></tr><tr><td>Total revenues</td><td>7,095</td><td>8,755</td><td>-1,660</td><td>-19</td></tr><tr><td>Banking and deposit interest expense</td><td>179</td><td>249</td><td>-70</td><td>-28</td></tr><tr><td>Total net revenues</td><td>6,916</td><td>8,506</td><td>-1,590</td><td>-19</td></tr><tr><td> Expenses</td><td></td><td></td><td></td><td></td></tr><tr><td>Distribution expenses</td><td>1,912</td><td>2,011</td><td>-99</td><td>-5</td></tr><tr><td>Interest credited to fixed accounts</td><td>790</td><td>847</td><td>-57</td><td>-7</td></tr><tr><td>Benefits, claims, losses and settlement expenses</td><td>1,125</td><td>1,179</td><td>-54</td><td>-5</td></tr><tr><td>Amortization of deferred acquisition costs</td><td>933</td><td>551</td><td>382</td><td>69</td></tr><tr><td>Interest and debt expense</td><td>109</td><td>112</td><td>-3</td><td>-3</td></tr><tr><td>Separation costs</td><td>\u2014</td><td>236</td><td>-236</td><td>-100</td></tr><tr><td>General and administrative expense</td><td>2,472</td><td>2,562</td><td>-90</td><td>-4</td></tr><tr><td>Total expenses</td><td>7,341</td><td>7,498</td><td>-157</td><td>-2</td></tr><tr><td>Pretax income (loss)</td><td>-425</td><td>1,008</td><td>-1,433</td><td>NM</td></tr><tr><td>Income tax provision (benefit)</td><td>-333</td><td>202</td><td>-535</td><td>NM</td></tr><tr><td>Net income (loss)</td><td>-92</td><td>806</td><td>-898</td><td>NM</td></tr><tr><td>Less: Net loss attributable to noncontrolling interests</td><td>-54</td><td>-8</td><td>-46</td><td>NM</td></tr><tr><td>Net income (loss) attributable to Ameriprise Financial</td><td>$-38</td><td>$814</td><td>$-852</td><td>NM</td></tr></table>"
        ]
dfs = parse_html_tables(data)
# Example of input data (similar to your case but generalized)
# data = {
#     "year": [2007, 2006, 2005],
#     "group_retirement_products": {
#         "premiums": [446, 386, 351],
#         "investment_income": [2280, 2279, 2233],
#         "capital_gains_losses": [-451, -144, -67],
#         "total_revenues": [2275, 2521, 2517],
#         "operating_income": [696, 1017, 1055]
#     },
#     "individual_fixed_annuities": {
#         "premiums": [96, 122, 97],
#         "investment_income": [3664, 3581, 3346],
#         "capital_gains_losses": [-829, -257, -214],
#         "total_revenues": [2931, 3446, 3229],
#         "operating_income": [530, 1036, 858]
#     },
#     # You can add more categories similarly
# }

# Convert to DataFrame
# df = convert_to_dataframe(data)

import os
from plot_table_3 import render_mpl_table

#
for i, df in enumerate(dfs):
    render_mpl_table(
        df,
        savename=f"test_{i}.png",
        # col_width=4.0,
    )
# render_mpl_table(
#     df,
#     # savename=os.path.join(img_base_dir, image_name),
#     savename='test.png',
#     # col_width=4.0,
# )
# import matplotlib.pyplot as plt
# import pandas as pd
# def plot_table(df, title):
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(8, 4))  # Adjust size according to your need
#     ax.set_axis_off()  # Hide the axis
#
#     # Create the table
#     table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
#                      cellLoc='center', loc='center', colColours=["#DCE3EC"] * df.shape[1])
#
#     # Style the table
#     table.auto_set_font_size(False)
#     table.set_fontsize(12)
#     table.scale(1.2, 1.2)  # Adjust scaling to fit the size
#
#     # Add title
#     plt.title(title, fontsize=14, weight='bold')
#
#     # Adjust layout and display
#     plt.tight_layout()
#     plt.show()
# plot_table(df, 'Generalized Financial Data')
# Display DataFrame
# import ace_tools as tools;

# tools.display_dataframe_to_user(name="Generalized Financial Data", dataframe=df)
#
