"""Phase 0 verification script. Run: python -m backend.verify_setup"""
import sys, os

def check(name, fn):
    try:
        fn()
        print(f"  ✅ {name}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False

def main():
    print("\n🧪 AMIA Phase 0 Verification\n" + "=" * 50)
    ok = True

    # Python version
    ok &= check("Python 3.10+", lambda: (
        assert_true(sys.version_info >= (3, 10), f"Got {sys.version}")
    ))

    # Virtual environment active
    ok &= check("Virtual environment active", lambda: (
        assert_true(sys.prefix != sys.base_prefix, "Run: source venv/bin/activate")
    ))

    # .env exists
    ok &= check(".env file exists", lambda: (
        assert_true(os.path.exists(".env"), "Create .env in project root")
    ))

    # Core imports
    for pkg_name, import_name in [
        ("LangGraph", "langgraph"),
        ("LangChain", "langchain"),
        ("ChromaDB", "chromadb"),
        ("FastAPI", "fastapi"),
        ("python-dotenv", "dotenv"),
    ]:
        ok &= check(f"{pkg_name} installed", lambda n=import_name: __import__(n))

    # API key present
    from dotenv import load_dotenv
    load_dotenv()
    ok &= check("At least one API key set", lambda: assert_true(
        (os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-") or
         os.getenv("OPENAI_API_KEY", "").startswith("sk-")),
        "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
    ))

    print("\n" + ("🎉 All checks passed!" if ok else "❌ Fix errors above."))
    return ok

def assert_true(condition, msg=""):
    if not condition:
        raise AssertionError(msg)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)