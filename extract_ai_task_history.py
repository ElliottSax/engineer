#!/usr/bin/env python3
"""
Extract AI Task History from Filmora's AITaskInfos.db
This database likely contains previous AI generation requests with API details!
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

def explore_ai_tasks_db():
    db_path = Path("/mnt/c/Users/ellio/AppData/Roaming/Wondershare/Wondershare Filmora/AICache/AITaskInfos.db")

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return

    print("=" * 80)
    print("FILMORA AI TASK HISTORY DATABASE")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("TABLES FOUND:")
    for table in tables:
        print(f"  - {table[0]}")
    print()

    # Explore each table
    for table in tables:
        table_name = table[0]

        print("=" * 80)
        print(f"TABLE: {table_name}")
        print("=" * 80)

        # Get schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        print("\nColumns:")
        for col in columns:
            print(f"  {col[1]:20s} ({col[2]})")

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"\nTotal rows: {count}")

        # Show all data if not too many rows
        if count > 0 and count <= 100:
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()

            print(f"\nAll {count} rows:\n")

            for row_idx, row in enumerate(rows, 1):
                print(f"Row {row_idx}:")
                for col_idx, col_info in enumerate(columns):
                    col_name = col_info[1]
                    value = row[col_idx]

                    # Try to parse JSON if it looks like JSON
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            parsed = json.loads(value)
                            value = json.dumps(parsed, indent=2)[:500]
                        except:
                            pass

                    # Truncate long values
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."

                    print(f"  {col_name}: {value}")
                print()

        print()

    conn.close()

    # Try to extract API-related info
    print("=" * 80)
    print("EXTRACTING API-RELATED INFORMATION")
    print("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Try common column patterns
    api_info = {
        "task_ids": [],
        "api_endpoints": [],
        "requests": [],
        "responses": []
    }

    for table in tables:
        table_name = table[0]

        try:
            # Get all data
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()

            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [col[1] for col in cursor.fetchall()]

            for row in rows:
                row_dict = dict(zip(columns, row))

                for key, value in row_dict.items():
                    if value and isinstance(value, str):
                        key_lower = key.lower()

                        # Look for task IDs
                        if 'task' in key_lower and 'id' in key_lower:
                            api_info["task_ids"].append(value)

                        # Look for URLs/endpoints
                        if 'url' in key_lower or 'endpoint' in key_lower:
                            api_info["api_endpoints"].append(value)

                        # Look for requests
                        if 'request' in key_lower:
                            try:
                                parsed = json.loads(value)
                                api_info["requests"].append(parsed)
                            except:
                                api_info["requests"].append(value)

                        # Look for responses
                        if 'response' in key_lower:
                            try:
                                parsed = json.loads(value)
                                api_info["responses"].append(parsed)
                            except:
                                api_info["responses"].append(value)

        except Exception as e:
            print(f"Error processing table {table_name}: {e}")

    conn.close()

    # Print summary
    print(f"\nTask IDs found: {len(api_info['task_ids'])}")
    for task_id in api_info['task_ids'][:10]:
        print(f"  {task_id}")

    print(f"\nAPI Endpoints found: {len(api_info['api_endpoints'])}")
    for endpoint in api_info['api_endpoints'][:10]:
        print(f"  {endpoint}")

    print(f"\nRequests found: {len(api_info['requests'])}")
    for req in api_info['requests'][:3]:
        print(f"  {json.dumps(req, indent=2)[:200]}...")

    print(f"\nResponses found: {len(api_info['responses'])}")
    for resp in api_info['responses'][:3]:
        print(f"  {json.dumps(resp, indent=2)[:200]}...")

    # Save to file
    output_file = Path("/mnt/e/wondershare/engineer/ai_task_history.json")
    with open(output_file, 'w') as f:
        json.dump(api_info, f, indent=2)

    print(f"\n✅ API info saved to: {output_file}")

if __name__ == "__main__":
    explore_ai_tasks_db()
