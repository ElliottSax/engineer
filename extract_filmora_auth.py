#!/usr/bin/env python3
"""
Extract Filmora Authentication Information
For use with your own paid Filmora subscription
"""

import sqlite3
import json
from pathlib import Path

def explore_database(db_path: Path):
    """Explore the Filmora authorization database."""
    print(f"Analyzing database: {db_path}\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("=" * 60)
    print("TABLES FOUND:")
    print("=" * 60)
    for table in tables:
        print(f"  - {table[0]}")
    print()

    # Explore each table
    for table in tables:
        table_name = table[0]
        print("=" * 60)
        print(f"TABLE: {table_name}")
        print("=" * 60)

        # Get schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        print("\nColumns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"\nRow count: {count}")

        # Show sample data (first 3 rows)
        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            rows = cursor.fetchall()

            print("\nSample data:")
            for i, row in enumerate(rows, 1):
                print(f"\n  Row {i}:")
                for col_idx, col_info in enumerate(columns):
                    col_name = col_info[1]
                    value = row[col_idx]

                    # Truncate long values
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."

                    print(f"    {col_name}: {value}")

        print()

    conn.close()

def extract_auth_info(db_path: Path) -> dict:
    """Extract authentication information from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    auth_info = {
        "tokens": [],
        "api_keys": [],
        "endpoints": [],
        "user_info": {}
    }

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    # Search for auth-related data
    for table in tables:
        cursor.execute(f"SELECT * FROM {table};")
        rows = cursor.fetchall()

        cursor.execute(f"PRAGMA table_info({table});")
        columns = [col[1] for col in cursor.fetchall()]

        for row in rows:
            row_dict = dict(zip(columns, row))

            # Look for tokens, keys, endpoints
            for key, value in row_dict.items():
                if value and isinstance(value, str):
                    key_lower = key.lower()

                    if 'token' in key_lower or 'auth' in key_lower:
                        auth_info["tokens"].append({
                            "table": table,
                            "column": key,
                            "value": value
                        })
                    elif 'key' in key_lower and 'api' in key_lower:
                        auth_info["api_keys"].append({
                            "table": table,
                            "column": key,
                            "value": value
                        })
                    elif 'endpoint' in key_lower or 'url' in key_lower or 'domain' in key_lower:
                        auth_info["endpoints"].append({
                            "table": table,
                            "column": key,
                            "value": value
                        })
                    elif 'user' in key_lower or 'account' in key_lower or 'email' in key_lower:
                        auth_info["user_info"][key] = value

    conn.close()
    return auth_info

def main():
    db_path = Path("/mnt/e/wondershare/Wondershare/Wondershare Filmora/15.0.12.16430/authorizeInfoCacheFile.db")

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    print("\n" + "=" * 60)
    print("FILMORA AUTHENTICATION DATABASE EXPLORER")
    print("=" * 60)
    print()

    # Explore database structure
    explore_database(db_path)

    # Extract auth info
    print("\n" + "=" * 60)
    print("EXTRACTED AUTHENTICATION INFORMATION")
    print("=" * 60)
    print()

    auth_info = extract_auth_info(db_path)

    # Save to JSON file
    output_path = Path("/mnt/e/wondershare/engineer/filmora_auth_info.json")
    with open(output_path, 'w') as f:
        json.dump(auth_info, f, indent=2)

    print(f"Authentication info saved to: {output_path}")
    print()
    print("Summary:")
    print(f"  - Tokens found: {len(auth_info['tokens'])}")
    print(f"  - API keys found: {len(auth_info['api_keys'])}")
    print(f"  - Endpoints found: {len(auth_info['endpoints'])}")
    print(f"  - User info fields: {len(auth_info['user_info'])}")

    # Show sample
    if auth_info['tokens']:
        print("\nSample token (first 50 chars):")
        token = auth_info['tokens'][0]['value']
        print(f"  {token[:50]}...")

    if auth_info['endpoints']:
        print("\nEndpoints found:")
        for endpoint in auth_info['endpoints'][:5]:
            print(f"  {endpoint['value']}")

if __name__ == "__main__":
    main()
