"""
Evaluates Docker build context size by applying .dockerignore rules.
Reports total uncompressed context bytes, file count, and large file breakdown.
"""

from pathlib import Path
import fnmatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_dockerignore(filepath: Path) -> list[str]:
    """Parse .dockerignore rules."""
    patterns = []
    if not filepath.is_file():
        return patterns
    for line in filepath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def is_ignored(rel_path: str, patterns: list[str]) -> bool:
    """Check if relative path matches any dockerignore pattern."""
    rel_path_fwd = rel_path.replace("\\", "/")
    ignored = False
    for pat in patterns:
        pat_clean = pat.rstrip("/")
        # Check direct match or directory prefix match
        if fnmatch.fnmatch(rel_path_fwd, pat) or fnmatch.fnmatch(rel_path_fwd, pat_clean) or rel_path_fwd.startswith(pat_clean + "/"):
            ignored = True
        elif pat.startswith("!"):
            unignore_pat = pat[1:].rstrip("/")
            if fnmatch.fnmatch(rel_path_fwd, unignore_pat) or rel_path_fwd.startswith(unignore_pat + "/"):
                ignored = False
    return ignored


def analyze_build_context():
    dockerignore_path = PROJECT_ROOT / ".dockerignore"
    patterns = parse_dockerignore(dockerignore_path)

    total_bytes = 0
    total_files = 0
    included_files = []

    for item in PROJECT_ROOT.rglob("*"):
        if item.is_file():
            rel_str = str(item.relative_to(PROJECT_ROOT))
            if not is_ignored(rel_str, patterns):
                file_size = item.stat().st_size
                total_bytes += file_size
                total_files += 1
                included_files.append((rel_str, file_size))

    print("=" * 60)
    print("DOCKER BUILD CONTEXT ANALYSIS")
    print("=" * 60)
    print(f"Total Included Files: {total_files}")
    print(f"Total Uncompressed Context Size: {total_bytes / (1024 * 1024):.2f} MB")
    print("-" * 60)
    print("Top 10 Largest Files in Context:")
    included_files.sort(key=lambda x: x[1], reverse=True)
    for name, sz in included_files[:10]:
        print(f"  {sz / (1024 * 1024):6.2f} MB : {name}")
    print("=" * 60)


if __name__ == "__main__":
    analyze_build_context()
