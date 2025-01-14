from typing import Any, Dict, List, Optional
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import snowflake.connector
import requests
import pandas as pd
from snowflake.core import Root
import generate_jwt
from dotenv import load_dotenv
import json
import io
import matplotlib
import matplotlib.pyplot as plt 
import time
from openai import OpenAI

matplotlib.use('Agg')
load_dotenv()

USER = os.getenv("USERNAME")
ACCOUNT = os.getenv("ACCOUNT")
WAREHOUSE = os.getenv("WAREHOUSE")
DATABASE = os.getenv("DATABASE")
SCHEMA = os.getenv("SCHEMA")
PASSWORD = os.getenv("PASSWORD")
ANALYST_ENDPOINT = os.getenv("ANALYST_ENDPOINT")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
RSA_PRIVATE_KEY_PATH = os.getenv("RSA_PRIVATE_KEY_PATH")
STAGE = os.getenv("SEMANTIC_MODEL_STAGE")
FILE = os.getenv("SEMANTIC_MODEL_FILE")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
ENABLE_CHARTS = True
DEBUG = False

openai = OpenAI()

# Initializes app
app = App(token=SLACK_BOT_TOKEN)
messages = []

@app.message("hello")
def message_hello(message, say):
    say(f"Hey there <@{message['user']}>!", thread_ts=message['ts'])
    say(
        text = "Let's BUILD",
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f":snowflake: Let's BUILD!",
                }
            },
        ],
        thread_ts=message['ts']
    )

@app.event("message")
def handle_message_events(ack, body, say):
    ack()
    prompt = body['event']['text']
    process_analyst_message(prompt, say, thread_ts=body['event']['ts'])

@app.command("/asksnowflake")
def ask_cortex(ack, body, say):
    ack()
    prompt = body['text']
    process_analyst_message(prompt, say, thread_ts=body['event']['ts'])

def process_analyst_message(prompt, say, thread_ts) -> Any:
    say_question(prompt, say, thread_ts)
    response = query_cortex_analyst(prompt)
    content = response["message"]["content"]
    display_analyst_content(content, say, thread_ts)

def say_question(prompt,say, thread_ts):
    say(
        text = "Question:",
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Question: {prompt}\nIf a date is provided, put it in the first column",
                }
            },
        ],
        thread_ts=thread_ts
    )
    say(
        text = "Snowflake Cortex Analyst is generating a response",
        blocks=[
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": "Snowflake Cortex Analyst is generating a response. Please wait...",
                }
            },
            {
                "type": "divider"
            },
        ],
        thread_ts=thread_ts
    )

def query_cortex_analyst(prompt) -> Dict[str, Any]:
    request_body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        # "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
        "semantic_model_file": f"@ALCHEMY_DEV.DEV_BLAKE.SEMANTIC_MODELS/{FILE}",
    }
    if DEBUG:
        print(request_body)
    resp = requests.post(
        url=f"{ANALYST_ENDPOINT}",
        json=request_body,
        headers={
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {JWT}",
        },
    )
    request_id = resp.headers.get("X-Snowflake-Request-Id")
    if resp.status_code == 200:
        if DEBUG:
            print(resp.text)
        return {**resp.json(), "request_id": request_id}  
    else:
        raise Exception(
            f"Failed request (id: {request_id}) with status {resp.status_code}: {resp.text}"
        )

def display_analyst_content(
    content: List[Dict[str, str]],
    say=None,
    thread_ts=None
) -> None:
    if DEBUG:
        print(content)
    for item in content:
        if item["type"] == "sql":
            say(
                text = "Generated SQL",
                blocks = [
                    {
                        "type": "rich_text",
                        "elements": [
                            {
                                "type": "rich_text_preformatted",
                                "elements": [
                                    {
                                        "type": "text",
                                        "text": f"{item['statement']}"
                                    }
                                ]
                            }
                        ]
                    }
                ],
                thread_ts=thread_ts
            )
            df = pd.read_sql(item["statement"], CONN)
            # Check if DataFrame size exceeds 50KB
            df_size = df.memory_usage(deep=True).sum()
            if df_size > 50 * 1024:  # 50KB in bytes
                # Split into chunks of 50 rows each
                chunks = [df[i:i+50] for i in range(0, len(df), 50)]
                for i, chunk in enumerate(chunks, 1):
                    say(
                        text = f"Answer (Part {i}/{len(chunks)}):",
                        blocks=[
                            {
                                "type": "rich_text",
                                "elements": [
                                    {
                                        "type": "rich_text_quote",
                                        "elements": [
                                            {
                                                "type": "text",
                                                "text": f"Answer (Part {i}/{len(chunks)}):",
                                                "style": {
                                                    "bold": True
                                                }
                                            }
                                        ]
                                    },
                                    {
                                        "type": "rich_text_preformatted",
                                        "elements": [
                                            {
                                                "type": "text",
                                                "text": f"{chunk.to_string()}"
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    )
            else:
                say(
                    text = "Answer:",
                    blocks=[
                        {
                            "type": "rich_text",
                            "elements": [
                                {
                                    "type": "rich_text_quote",
                                    "elements": [
                                        {
                                            "type": "text",
                                            "text": "Answer:",
                                            "style": {
                                                "bold": True
                                            }
                                        }
                                    ]
                                },
                                {
                                    "type": "rich_text_preformatted",
                                    "elements": [
                                        {
                                            "type": "text",
                                            "text": f"{df.to_string()}"
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                    thread_ts=thread_ts
                )                    
            if ENABLE_CHARTS and len(df.columns) > 1:
                chart_img_url = plot_chart(df)
                if chart_img_url is not None:
                    say(
                        text = "Chart",
                        blocks=[
                            {
                                "type": "image",
                                "title": {
                                    "type": "plain_text",
                                    "text": "Chart"
                                },
                                "block_id": "image",
                                "slack_file": {
                                    "url": f"{chart_img_url}"
                                },
                                "alt_text": "Chart"
                            }
                        ],
                        thread_ts=thread_ts
                    )
        elif item["type"] == "text":
            say(
                text = "Answer:",
                blocks = [
                    {
                        "type": "rich_text",
                        "elements": [
                            {
                                "type": "rich_text_quote",
                                "elements": [
                                    {
                                        "type": "text",
                                        "text": f"{item['text']}"
                                    }
                                ]
                            }
                        ]
                    }
                ],
                thread_ts=thread_ts
            )
        elif item["type"] == "suggestions":
            suggestions = "You may try these suggested questions: \n\n- " + "\n- ".join(item['suggestions']) + "\n\nNOTE: There's a 150 char limit on Slack messages so alter the questions accordingly."
            say(
                text = "Suggestions:",
                blocks = [
                    {
                        "type": "rich_text",
                        "elements": [
                            {
                                "type": "rich_text_preformatted",
                                "elements": [
                                    {
                                        "type": "text",
                                        "text": f"{suggestions}"
                                    }
                                ]
                            }
                        ]
                    }
                ],
                thread_ts=thread_ts
            )               

def query_llm_for_chart(df) -> Any:
    # Convert DataFrame to a string representation for the API
    # df_str = df.head().to_string()

    # cortex_payload = {
    #     "model": "llama3.1-8b",  # Replace with your actual model name
    #     "messages": [
    #         {"role": "system", "content": "You are a data visualization assistant."},
    #         {"role": "user", "content": f"The following is a sample of a DataFrame:\n{df_str}\nBased on this data, determine the best chart type (line, bar, or pie, or none). Respond with one word specifying the type."}
    #     ]
    # }

    # resp = requests.post(
    #     url=f"{LLM_ENDPOINT}",
    #     json=cortex_payload,
    #     headers={
    #         "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
    #         "Content-Type": "application/json",
    #         "Accept": "application/json",
    #         "Authorization": f"Bearer {JWT}",
    #     },
    # )
    # if resp.status_code == 200:
    #     return resp.text
    # else:
    #     raise ValueError(f"Cortex API call failed: {resp.status_code} {resp.text}")

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user", 
                "content": f"The following is a sample of a DataFrame:\n{df}\n Based on the data, chart the data in the optimal way using matplotlib and return just the python code to generate the chart, no other text whatsoever. You can assume you have access to the df variable in the code. Don't show the plot, I'll handle the display."
            }
        ]
    )

    code = response.choices[0].message.content

    return code

def plot_chart(df):
    # Try to detect if the data represents a time series
    is_time_series = False

    response = query_llm_for_chart(df)
    cleaned_response = response.strip('```python').strip('```').strip()
    exec(cleaned_response)
    # print(response)
    # chart_type = None
    # if "bar" in response.lower():
    #     chart_type = "bar"
    # elif "pie" in response.lower():
    #     chart_type = "pie"
    # elif "line" in response.lower():
    #     chart_type = "line"
    # else:
    #     raise ValueError("Unsupported chart type")

    # # Check if the index is datetime-like
    # if pd.api.types.is_datetime64_any_dtype(df.index):
    #     is_time_series = True
    # else:
    #     # Attempt to convert the first column to datetime
    #     try:
    #         df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    #         df.set_index(df.columns[0], inplace=True)
    #         is_time_series = True
    #     except (ValueError, TypeError):
    #         pass

    # # Determine the chart type
    # if is_time_series:
    #     chart_type = 'line'
    # elif len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
    #     chart_type = 'pie'
    # else:
    #     raise ValueError("Unsupported DataFrame structure for plotting.")

    # plt.figure(figsize=(10, 6), facecolor='#333333')

    # if chart_type == 'line':
    #     # Line chart
    #     for col in df.columns:
    #         plt.plot(df.index, df[col], label=col)
    #     plt.xlabel("Time", fontsize=14, color="white")
    #     plt.ylabel("Values", fontsize=14, color="white")
    #     plt.title("Time Series Line Chart", fontsize=16, color="white")
    #     plt.legend(fontsize=12, loc='best')  # Add a legend for line chart
    #     plt.gca().set_facecolor('#333333')
    #     plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # elif chart_type == 'bar':
    #     # Bar chart
    #     plt.bar(df.index, df[df.columns[0]], color=plt.cm.tab20.colors[0])
    #     plt.xlabel(df.index.name or "Categories", fontsize=14, color="white")
    #     plt.ylabel("Values", fontsize=14, color="white")
    #     plt.title("Bar Chart", fontsize=16, color="white")
    #     plt.xticks(rotation=45)
    #     plt.gca().set_facecolor('#333333')
    #     plt.grid(color='gray', linestyle='--', linewidth=0.5, axis='y')

    # elif chart_type == 'pie':
    #     # Pie chart
    #     wedges, texts, autotexts = plt.pie(df[df.columns[1]], 
    #                                     # labels=df[df.columns[0]], 
    #                                     autopct='%1.1f%%', 
    #                                     startangle=90, 
    #                                     colors=plt.cm.tab20.colors[:len(df)], 
    #                                     textprops={'color': "white", 'fontsize': 16})
    #     plt.axis('equal')  # Equal aspect ratio for pie chart
    #     plt.gca().set_facecolor('#333333')

    #     # Add a legend for the pie chart
    #     plt.legend(wedges, df[df.columns[0]], title="Legend", fontsize=12, title_fontsize=14, loc='best', bbox_to_anchor=(1, 0.5))


    # Save the chart as a .jpg file
    plt.tight_layout()
    file_path_jpg = 'chart.jpg'
    plt.savefig(file_path_jpg, format='jpg')
    plt.close()  # Close the plot to free resources

    # Slack upload logic remains unchanged
    file_size = os.path.getsize(file_path_jpg)
    file_upload_url_response = app.client.files_getUploadURLExternal(filename=file_path_jpg, length=file_size)
    
    if DEBUG:
        print(file_upload_url_response)
    file_upload_url = file_upload_url_response['upload_url']
    file_id = file_upload_url_response['file_id']
    
    with open(file_path_jpg, 'rb') as f:
        response = requests.post(file_upload_url, files={'file': f})
    
    # Check the response
    img_url = None
    if response.status_code != 200:
        print("File upload failed", response.text)
    else:
        # Complete upload and get permalink to display
        response = app.client.files_completeUploadExternal(files=[{"id": file_id, "title": "chart"}])
        if DEBUG:
            print(response)
        img_url = response['files'][0]['permalink']
        time.sleep(2)
    
    return img_url



def init():
    conn,jwt = None,None
    conn = snowflake.connector.connect(
        user=USER,
        password=PASSWORD,
        account=ACCOUNT,
        warehouse=WAREHOUSE
    )
    
    jwt = generate_jwt.JWTGenerator(ACCOUNT,USER,RSA_PRIVATE_KEY_PATH).get_token()
    return conn,jwt

# Start app
if __name__ == "__main__":
    CONN,JWT = init()
    if not CONN.rest.token:
        print("Error: Failed to connect to Snowflake! Please check your Snowflake user, password, and account environment variables and try again.")
        quit()
    Root = Root(CONN)
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
    