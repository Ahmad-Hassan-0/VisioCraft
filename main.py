"""
VisioCraft AI - Main Entry Point
"""

import sys
import webbrowser
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # Add project root to path

import config


def open_browser():
    """Open the web browser after a short delay"""
    time.sleep(2)  # Wait for server to start
    url = f"http://localhost:{config.SERVER_PORT}"
    print(f"\n{'='*60}")
    print(f"🚀 VisioCraft AI is running!")
    print(f"📱 Access the application at: {url}")
    print(f"{'='*60}\n")
    webbrowser.open(url)


def main():
    """Main entry point for the application"""
    print("""
    VisioCraft AI
    """)

    print("🔧 Initializing server components...")

    # Create the app via the factory
    from app import create_app
    application = create_app()

    print("✅ Server initialized successfully")
    print(f"📂 Output directory: {config.OUTPUT_DIR}")
    print(f"🔌 Starting server on port {config.SERVER_PORT}...")

    # Start browser in separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Run Flask server
    application.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=config.DEBUG_MODE,
        use_reloader=False,  # Disable reloader to prevent double initialization
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down VisioCraft AI. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)