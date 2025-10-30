"""Test script to verify OpenAI API key is working."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_key():
    """Test if OpenAI API key is configured and working."""
    
    print("=" * 80)
    print("OpenAI API Key Test")
    print("=" * 80)
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ FAILED: OPENAI_API_KEY not found in environment variables")
        print("\nPlease ensure your .env file contains:")
        print("OPENAI_API_KEY=sk-...")
        return False
    
    # Mask the key for display
    if api_key.startswith("sk-"):
        masked_key = f"{api_key[:7]}...{api_key[-4:]}"
    else:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    
    print(f"✅ API Key found: {masked_key}")
    print(f"   Key length: {len(api_key)} characters")
    
    # Try to import OpenAI
    try:
        from openai import OpenAI
        print("✅ OpenAI library imported successfully")
    except ImportError as e:
        print(f"❌ FAILED: OpenAI library not installed")
        print(f"   Error: {e}")
        print("\nPlease install it with: pip install openai")
        return False
    
    # Try to initialize client
    try:
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"❌ FAILED: Could not initialize OpenAI client")
        print(f"   Error: {e}")
        return False
    
    # Try to make a simple API call
    print("\n🔄 Testing API connection with a simple call...")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API test successful' in exactly 3 words."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content
        print(f"✅ API call successful!")
        print(f"   Model: {response.model}")
        print(f"   Response: {result}")
        print(f"   Tokens used: {response.usage.total_tokens}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - OpenAI API key is working correctly!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"❌ FAILED: API call failed")
        print(f"   Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Provide specific guidance based on error
        error_str = str(e).lower()
        if "authentication" in error_str or "invalid" in error_str or "401" in error_str:
            print("\n💡 This looks like an authentication error.")
            print("   - Check that your API key is correct and active")
            print("   - Verify it starts with 'sk-'")
            print("   - Make sure you haven't exceeded your API quota")
            print("   - Check https://platform.openai.com/account/api-keys")
        elif "rate limit" in error_str or "429" in error_str:
            print("\n💡 Rate limit exceeded.")
            print("   - Wait a moment and try again")
            print("   - Check your API usage limits")
        elif "connection" in error_str or "network" in error_str:
            print("\n💡 Network connection issue.")
            print("   - Check your internet connection")
            print("   - Verify firewall settings")
        
        print("\n" + "=" * 80)
        print("❌ TEST FAILED - Please fix the issues above")
        print("=" * 80)
        return False


if __name__ == "__main__":
    try:
        success = test_openai_key()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

