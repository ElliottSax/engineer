#!/usr/bin/env python3
"""
Decode Filmora Authentication Data
The INFO field appears to be hex-encoded
"""

import sqlite3
import json
import base64
from pathlib import Path

def extract_and_decode():
    db_path = Path("/mnt/e/wondershare/Wondershare/Wondershare Filmora/15.0.12.16430/authorizeInfoCacheFile.db")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the authorize info
    cursor.execute("SELECT INFO FROM AppSetting WHERE INFONAME='Authorize';")
    result = cursor.fetchone()

    if not result:
        print("No authorization data found")
        return

    encoded_data = result[0]

    print("=" * 60)
    print("FILMORA AUTHORIZATION DATA")
    print("=" * 60)
    print()
    print(f"Raw data length: {len(encoded_data)} characters")
    print(f"First 100 chars: {encoded_data[:100]}")
    print()

    # Try hex decoding
    try:
        hex_decoded = bytes.fromhex(encoded_data)
        print("✓ Successfully hex decoded")
        print(f"Decoded length: {len(hex_decoded)} bytes")
        print()

        # Try to find readable strings
        text = hex_decoded.decode('utf-8', errors='ignore')
        print("Decoded text (with errors ignored):")
        print("=" * 60)
        print(text[:1000])  # First 1000 chars
        print("=" * 60)
        print()

        # Save raw decoded data
        output_path = Path("/mnt/e/wondershare/engineer/filmora_auth_decoded.bin")
        with open(output_path, 'wb') as f:
            f.write(hex_decoded)
        print(f"✓ Raw decoded data saved to: {output_path}")

        # Save text version
        text_path = Path("/mnt/e/wondershare/engineer/filmora_auth_decoded.txt")
        with open(text_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(text)
        print(f"✓ Text version saved to: {text_path}")
        print()

        # Look for JSON structures
        if '{' in text and '}' in text:
            print("Found potential JSON structures, extracting...")
            # Find JSON objects
            start_indices = [i for i, char in enumerate(text) if char == '{']
            for start in start_indices[:5]:  # Check first 5 occurrences
                for end in range(start + 1, min(start + 5000, len(text))):
                    if text[end] == '}':
                        try:
                            potential_json = text[start:end+1]
                            parsed = json.loads(potential_json)
                            print(f"\n✓ Found valid JSON at position {start}:")
                            print(json.dumps(parsed, indent=2)[:500])
                            break
                        except:
                            continue

        # Look for URLs/endpoints
        if 'http' in text.lower():
            print("\n✓ Found HTTP endpoints:")
            lines = text.split('\n')
            for line in lines:
                if 'http' in line.lower():
                    print(f"  {line.strip()[:200]}")

        # Look for tokens (long alphanumeric strings)
        import re
        tokens = re.findall(r'[a-zA-Z0-9_-]{40,}', text)
        if tokens:
            print(f"\n✓ Found {len(tokens)} potential tokens/keys:")
            for token in tokens[:5]:
                print(f"  {token[:80]}...")

    except Exception as e:
        print(f"✗ Error decoding: {e}")

        # Try alternative: might be base64
        try:
            b64_decoded = base64.b64decode(encoded_data)
            print("\n✓ Successfully base64 decoded")
            text = b64_decoded.decode('utf-8', errors='ignore')
            print(text[:500])
        except:
            print("✗ Base64 decoding also failed")

    conn.close()

if __name__ == "__main__":
    extract_and_decode()
