#!/usr/bin/env python3
"""
Test the complete Filmora integration system
Verifies all components are working
"""

import asyncio
import json
import sys
import os
from pathlib import Path
import time

# Add imports
sys.path.append(str(Path(__file__).parent / "hybrid_pipeline"))

print("=" * 60)
print("🧪 FILMORA INTEGRATION SYSTEM TEST")
print("=" * 60)

def test_imports():
    """Test all required imports"""
    print("\n1️⃣ Testing imports...")

    try:
        from filmora_api_interface import FilmoraAPIClient, FilmoraAuth
        print("  ✅ filmora_api_interface")
    except Exception as e:
        print(f"  ❌ filmora_api_interface: {e}")
        return False

    try:
        from ultra_low_cost_pipeline import UltraLowCostVideoPipeline
        print("  ✅ ultra_low_cost_pipeline")
    except Exception as e:
        print(f"  ❌ ultra_low_cost_pipeline: {e}")
        return False

    try:
        from cheap_api_router import CheapAPIRouter
        print("  ✅ cheap_api_router")
    except Exception as e:
        print(f"  ❌ cheap_api_router: {e}")
        return False

    try:
        from filmora_zero_cost_pipeline import FilmoraZeroCostPipeline
        print("  ✅ filmora_zero_cost_pipeline")
    except Exception as e:
        print(f"  ❌ filmora_zero_cost_pipeline: {e}")
        return False

    return True

def test_captured_auth():
    """Test if authentication has been captured"""
    print("\n2️⃣ Testing captured authentication...")

    capture_dir = Path("captured_api_data")

    if not capture_dir.exists():
        print("  ❌ No captured data directory")
        return None

    captures = list(capture_dir.glob("*.json"))

    if not captures:
        print("  ❌ No capture files found")
        return None

    # Get latest capture
    latest = max(captures, key=lambda p: p.stat().st_mtime)
    print(f"  📂 Found capture: {latest.name}")

    with open(latest, 'r') as f:
        data = json.load(f)

    auth = data.get("auth", {})

    # Check required fields
    required = ["client_sign", "base_api"]
    optional = ["wsid", "session_token", "device_id"]

    for field in required:
        if field in auth and auth[field]:
            print(f"  ✅ {field}: {str(auth[field])[:30]}...")
        else:
            print(f"  ❌ {field}: Missing")
            return None

    for field in optional:
        if field in auth and auth[field]:
            print(f"  ✅ {field}: {str(auth[field])[:30]}...")
        else:
            print(f"  ⚠️ {field}: Not captured yet")

    return auth

async def test_filmora_api(auth_data):
    """Test Filmora API connection"""
    print("\n3️⃣ Testing Filmora API...")

    if not auth_data.get("wsid"):
        print("  ⚠️ Cannot test - wsid not captured")
        print("  Run capture_filmora_live.py while using Filmora")
        return False

    from filmora_api_interface import FilmoraAuth, FilmoraAPIClient
    from datetime import datetime, timedelta

    # Create auth object
    filmora_auth = FilmoraAuth(
        user_id=auth_data.get("device_id", "test"),
        auth_token=auth_data.get("wsid"),
        refresh_token=auth_data.get("session_token", ""),
        subscription_type="active",
        api_endpoint=auth_data.get("base_api"),
        expires_at=datetime.now() + timedelta(days=30),
        device_id=auth_data.get("device_id", "")
    )

    # Add additional fields
    filmora_auth.client_sign = auth_data.get("client_sign")
    filmora_auth.product_id = auth_data.get("product_id", "1901")

    # Test API client
    client = FilmoraAPIClient(filmora_auth)

    try:
        async with client as c:
            # Test user info
            user_info = await c.get_user_info()
            if user_info.get("success"):
                print(f"  ✅ User authenticated")
                print(f"  ✅ Subscription: {user_info.get('subscription_type', 'Unknown')}")
            else:
                print(f"  ❌ Authentication failed")
                return False

            # Test capabilities
            capabilities = await c.get_capabilities()
            print(f"  ✅ Available features: {len(capabilities.get('features', []))}")

            return True
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False

def test_video_pipeline():
    """Test video generation pipeline"""
    print("\n4️⃣ Testing video pipeline...")

    try:
        from ultra_low_cost_pipeline import UltraLowCostVideoPipeline

        pipeline = UltraLowCostVideoPipeline()
        print("  ✅ Pipeline initialized")

        # Check cost estimates
        estimate = pipeline.estimate_video_cost(duration_minutes=6)
        print(f"  💰 Estimated cost for 6-min video: ${estimate:.2f}")

        if estimate < 0.5:
            print(f"  ✅ Under target cost ($0.50)")
        else:
            print(f"  ⚠️ Above target cost")

        return True

    except Exception as e:
        print(f"  ❌ Pipeline test failed: {e}")
        return False

def test_directories():
    """Test required directories exist"""
    print("\n5️⃣ Testing directories...")

    dirs = [
        "captured_api_data",
        "output",
        "temp",
        "hybrid_pipeline"
    ]

    all_exist = True

    for dir_name in dirs:
        if Path(dir_name).exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (creating...)")
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            all_exist = False

    return all_exist

async def run_complete_test():
    """Run complete system test"""

    results = {
        "imports": test_imports(),
        "directories": test_directories(),
        "auth": None,
        "api": False,
        "pipeline": test_video_pipeline()
    }

    # Test authentication
    auth_data = test_captured_auth()
    results["auth"] = auth_data is not None

    # Test API if auth available
    if auth_data and auth_data.get("wsid"):
        results["api"] = await test_filmora_api(auth_data)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("System ready for video generation!")
        print("\nRun: python run_hybrid_pipeline.py")
    elif results["imports"] and results["directories"] and results["pipeline"]:
        print("\n⚠️ PARTIAL SUCCESS")
        print("Core systems working, but authentication needed.")
        print("\nNext steps:")
        print("1. Run: python capture_filmora_live.py")
        print("2. Open Filmora and use an AI feature")
        print("3. Wait for wsid capture")
        print("4. Run this test again")
    else:
        print("\n❌ SYSTEM NOT READY")
        print("Please fix the errors above and try again")

    print("=" * 60)

async def main():
    """Main test entry point"""
    await run_complete_test()

if __name__ == "__main__":
    print("\n🚀 Starting system test...")
    print("This will verify all components are working")
    print("-" * 40)

    asyncio.run(main())