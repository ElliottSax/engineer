#!/usr/bin/env python3
"""
Test Cheap API Pipeline
Works without Filmora - uses ultra-low-cost APIs
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "hybrid_pipeline"))

from ultra_low_cost_pipeline import UltraLowCostVideoPipeline
from cheap_api_router import CheapAPIRouter, TaskType

print("=" * 60)
print("🚀 TESTING CHEAP API PIPELINE")
print("=" * 60)
print("This will test video generation with < $0.50 cost")
print("No Filmora required - uses free/cheap APIs only")
print("=" * 60)

async def test_pipeline():
    """Test the ultra-low-cost pipeline"""

    # Initialize pipeline
    pipeline = UltraLowCostVideoPipeline()

    print("\n📊 Available APIs:")
    print("  ✅ DeepSeek ($0.14/1M tokens)")
    print("  ✅ Google Gemini (FREE - 1000 images/day)")
    print("  ✅ HuggingFace (FREE models)")
    print("  ✅ Together AI ($0.20/1M tokens)")
    print("  ✅ Groq (FREE tier)")

    # Test topic
    topic = "How Social Media Algorithms Control Your Mind"

    print(f"\n🎬 Test Video: {topic}")
    print("Target: 6-minute explainer video")
    print("Style: Minimalist stick figures")

    # Estimate cost
    estimate = pipeline.estimate_video_cost(duration_minutes=6)
    print(f"\n💰 Estimated cost: ${estimate:.2f}")

    if estimate > 0.50:
        print("⚠️ Cost higher than target, optimizing...")

    print("\n" + "-" * 40)
    print("STARTING GENERATION")
    print("-" * 40)

    try:
        # Generate video
        result = await pipeline.generate_ultra_low_cost_video(
            topic=topic,
            duration_minutes=6,
            max_budget=0.50
        )

        print("\n" + "=" * 60)
        print("✅ VIDEO GENERATION COMPLETE!")
        print("=" * 60)
        print(f"📹 Output: {result['path']}")
        print(f"⏱️ Duration: {result['duration']} seconds")
        print(f"💰 Total Cost: ${result['cost']:.2f}")
        print(f"📊 Scenes: {result['scenes']}")

        # Cost breakdown
        if "cost_breakdown" in result:
            print("\nCost Breakdown:")
            for item, cost in result["cost_breakdown"].items():
                print(f"  {item}: ${cost:.3f}")

        print("=" * 60)

        return result

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check API keys in .env file")
        print("2. Ensure FFmpeg is installed")
        print("3. Check network connection")
        return None

async def test_router():
    """Test the cheap API router"""

    print("\n🔄 Testing API Router...")

    router = CheapAPIRouter()

    # Test script generation
    test_prompt = "Write a 30-second script about AI"

    result = await router.route_request(
        TaskType.SCRIPT_GENERATION,
        {"messages": [{"role": "user", "content": test_prompt}]},
        prefer_free=True
    )

    if result.get("success"):
        print("  ✅ Router working")
        print(f"  Provider: {result.get('provider', 'Unknown')}")
        print(f"  Cost: ${result.get('cost', 0):.4f}")
    else:
        print("  ❌ Router failed")
        print(f"  Error: {result.get('error', 'Unknown')}")

async def main():
    """Main test function"""

    print("\n1️⃣ Testing API Router...")
    await test_router()

    print("\n2️⃣ Testing Video Pipeline...")
    result = await test_pipeline()

    if result:
        print("\n🎉 SUCCESS!")
        print("The cheap API pipeline is working!")
        print("\nNext steps:")
        print("1. Capture Filmora wsid for zero-cost generation")
        print("2. Or continue using cheap APIs (< $0.50/video)")
    else:
        print("\n⚠️ Pipeline test failed")
        print("Check the error messages above")

if __name__ == "__main__":
    print("\n🚀 Starting cheap API test...")
    print("This doesn't require Filmora")
    print("-" * 40)

    asyncio.run(main())