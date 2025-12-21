#!/usr/bin/env python3
"""
Analyze existing Filmora logs for API patterns
Extract any AI-related calls that already happened
"""

import re
import json
from pathlib import Path
from collections import defaultdict

def analyze_logs():
    log_dir = Path("/mnt/e/wondershare/Wondershare/Wondershare Filmora/15.0.12.16430/log")

    # Find all main logs
    log_files = sorted(log_dir.glob("APP@MAIN*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not log_files:
        print("No log files found!")
        return

    print("=" * 80)
    print("ANALYZING FILMORA LOGS FOR API PATTERNS")
    print("=" * 80)
    print()

    api_calls = []
    endpoints = defaultdict(int)
    auth_info = set()

    for log_file in log_files[:3]:  # Analyze last 3 log files
        print(f"Analyzing: {log_file.name}")

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')

                # Extract all HTTP URLs
                urls = re.findall(r'https?://[^\s\'"]+', content)

                for url in urls:
                    # Clean up URL
                    url = url.rstrip(',')

                    # Track unique endpoints
                    base_endpoint = url.split('?')[0]
                    endpoints[base_endpoint] += 1

                    # Look for AI-related endpoints
                    if any(keyword in url.lower() for keyword in ['aigc', 'veo', 'sora', 'video', 'image']):
                        api_calls.append(url)

                # Extract request bodies
                req_bodies = re.findall(r'reqBody\s+({[^}]+})', content)

                # Extract client_sign patterns
                client_signs = re.findall(r'client_sign[=:]([^&\s\'"]+)', content)
                for sign in client_signs:
                    auth_info.add(f"client_sign: {sign}")

                # Extract wsid patterns
                wsids = re.findall(r'wsid[=:]([^&\s\'"]+)', content)
                for wsid in wsids:
                    if wsid and wsid != '""' and wsid != '':
                        auth_info.add(f"wsid: {wsid}")

        except Exception as e:
            print(f"Error reading {log_file.name}: {e}")

    print("\n" + "=" * 80)
    print("TOP ENDPOINTS USED")
    print("=" * 80)

    sorted_endpoints = sorted(endpoints.items(), key=lambda x: x[1], reverse=True)
    for endpoint, count in sorted_endpoints[:20]:
        print(f"{count:3d}x  {endpoint}")

    print("\n" + "=" * 80)
    print("AI-RELATED API CALLS")
    print("=" * 80)

    unique_ai_calls = list(set(api_calls))
    if unique_ai_calls:
        for call in unique_ai_calls[:10]:
            print(f"  {call[:120]}")
    else:
        print("  ⚠️  No AI API calls found in logs")
        print("  You need to use Filmora's AI features to capture the API calls")

    print("\n" + "=" * 80)
    print("AUTHENTICATION INFO FOUND")
    print("=" * 80)

    if auth_info:
        for info in sorted(auth_info)[:10]:
            print(f"  {info}")
    else:
        print("  ⚠️  No auth info found")

    # Save results
    output = {
        'endpoints': dict(sorted_endpoints),
        'ai_api_calls': unique_ai_calls,
        'auth_info': list(auth_info)
    }

    output_file = Path("/mnt/e/wondershare/engineer/log_analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    return output

if __name__ == "__main__":
    analyze_logs()
