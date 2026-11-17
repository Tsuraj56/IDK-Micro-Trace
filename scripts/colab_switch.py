"""
colab_switch.py - Multi-Account Profile and Token Switcher for Google Colab CLI
"""
import sys
import os
import re
import json
import base64
import shutil
import urllib.request
from typing import Optional
from pathlib import Path
from datetime import datetime
import keyring

KEYRING_SERVICE = "google-colab-cli"
KEYRING_USERNAME = "oauth_token"

DEFAULT_CLI_DIR = Path.home() / ".config" / "colab-cli"
PROFILES_DIR = Path.home() / ".config" / "colab_profiles"
ACTIVE_FILE = PROFILES_DIR / "active_profile.txt"


def get_active_file() -> Path:
    """Returns path to active profile file, respecting any override of PROFILES_DIR."""
    return PROFILES_DIR / "active_profile.txt"


def is_valid_profile_name(name: str) -> bool:
    """Ensures profile name is safe and contains only allowed characters."""
    if not name or len(name) > 64:
        return False
    return re.match(r"^[a-zA-Z0-9_\-]+$", name) is not None


def ensure_dirs():
    DEFAULT_CLI_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def get_active_profile_name() -> str:
    active_path = get_active_file()
    if active_path.exists():
        try:
            name = active_path.read_text(encoding="utf-8").strip()
            if name and is_valid_profile_name(name):
                return name
        except Exception:
            pass
    return "account1"


def set_active_profile_name(name: str):
    if not is_valid_profile_name(name):
        raise ValueError(f"Invalid profile name: '{name}'")
    ensure_dirs()
    get_active_file().write_text(name.strip(), encoding="utf-8")


def get_profile_dir(name: str) -> Path:
    if not is_valid_profile_name(name):
        raise ValueError(f"Invalid profile name: '{name}'")
    pdir = PROFILES_DIR / name.strip()
    pdir.mkdir(parents=True, exist_ok=True)
    return pdir


def is_valid_token_json(token_str: str) -> bool:
    """Checks whether token_str contains valid OAuth credentials."""
    if not token_str or not token_str.strip():
        return False
    try:
        data = json.loads(token_str)
        if not isinstance(data, dict):
            return False
        tok = data.get("token")
        ref = data.get("refresh_token")
        has_tok = isinstance(tok, str) and bool(tok.strip())
        has_ref = isinstance(ref, str) and bool(ref.strip())
        return has_tok or has_ref
    except Exception:
        return False


def extract_email_from_token(token_str: str) -> str:
    """Extracts email from token JSON if available in id_token or account field."""
    if not token_str:
        return ""
    try:
        data = json.loads(token_str)
        if not isinstance(data, dict):
            return ""
        # Check account field (from google.oauth2.credentials)
        acc = data.get("account")
        if acc and isinstance(acc, str) and "@" in acc:
            return acc.strip()

        # Check id_token JWT (if present)
        id_token = data.get("id_token")
        if id_token and isinstance(id_token, str):
            parts = id_token.split(".")
            if len(parts) >= 2:
                payload = parts[1]
                padded = payload + "=" * (-len(payload) % 4)
                claims = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
                if isinstance(claims, dict):
                    email = claims.get("email")
                    if email and isinstance(email, str) and "@" in email:
                        return str(email).strip()
    except Exception:
        pass
    return ""


def get_profile_email_info(profile_name: str, token_str: str = "") -> tuple[str, bool]:
    """
    Resolves the email address and verification status for a profile.
    Returns:
        (email, is_verified)
        - is_verified is True if the email was extracted from valid OAuth credentials.
        - is_verified is False if the email came from an explicit manual label (email.txt).
        - returns ("", False) if neither is available.
    """
    try:
        pdir = get_profile_dir(profile_name)
    except Exception:
        return ("", False)

    # 1. Check real token credentials first (verified source)
    if not token_str:
        token_file = pdir / "token.json"
        if token_file.exists():
            try:
                token_str = token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

    if token_str and is_valid_token_json(token_str):
        emb_email = extract_email_from_token(token_str)
        if emb_email:
            return (emb_email, True)

    # 2. Check explicit manual label in <profile_dir>/email.txt (unverified label)
    label_file = pdir / "email.txt"
    if label_file.exists():
        try:
            manual_email = label_file.read_text(encoding="utf-8").strip()
            if manual_email:
                return (manual_email, False)
        except Exception:
            pass

    return ("", False)


def resolve_email_for_profile(profile_name: str, token_str: str = "", allow_network: bool = False) -> str:
    """
    Resolves the email address string for a profile.
    Network requests are disabled. Returns email address string if found, or empty string.
    """
    email, _ = get_profile_email_info(profile_name, token_str=token_str)
    return email


def clear_active_credentials():
    """Clears both OS keyring and legacy token file so credentials never bleed across profiles."""
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:
        pass

    try:
        if keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) is not None:
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
            except Exception:
                pass
    except Exception:
        pass

    legacy_token = DEFAULT_CLI_DIR / "token.json"
    if legacy_token.exists():
        try:
            legacy_token.unlink()
        except Exception as e:
            print(f"[Warning] Failed to remove legacy token: {e}", file=sys.stderr)


def save_current_state_to_profile(profile_name: str) -> bool:
    """Saves active keyring token and active CLI sessions to profile dir."""
    if not is_valid_profile_name(profile_name):
        return False
    ensure_dirs()
    pdir = get_profile_dir(profile_name)

    # 1. Save keyring token if present
    token = None
    try:
        token = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception as e:
        print(f"[Warning] Failed to read token from keyring: {e}", file=sys.stderr)

    # Fallback to legacy token file if keyring had nothing
    legacy_token = DEFAULT_CLI_DIR / "token.json"
    if not token and legacy_token.exists():
        try:
            token = legacy_token.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    token_saved = False
    if token and is_valid_token_json(token):
        try:
            (pdir / "token.json").write_text(token, encoding="utf-8")
            token_saved = True
        except Exception as e:
            print(f"[Warning] Failed to write token to profile dir: {e}", file=sys.stderr)

    # 2. Copy sessions.json if present
    cli_sessions = DEFAULT_CLI_DIR / "sessions.json"
    if cli_sessions.exists():
        try:
            shutil.copy2(cli_sessions, pdir / "sessions.json")
        except Exception as e:
            print(f"[Warning] Failed to copy sessions.json: {e}", file=sys.stderr)

    # 3. Copy settings.json if present
    cli_settings = DEFAULT_CLI_DIR / "settings.json"
    if cli_settings.exists():
        try:
            shutil.copy2(cli_settings, pdir / "settings.json")
        except Exception:
            pass

    return token_saved


def load_profile_state(profile_name: str):
    """Restores profile token to keyring and session files to colab-cli dir."""
    if not is_valid_profile_name(profile_name):
        raise ValueError(f"Invalid profile name: '{profile_name}'. Must be alphanumeric with '-' or '_'.")
    ensure_dirs()
    pdir = get_profile_dir(profile_name)

    # Clear legacy token in DEFAULT_CLI_DIR first so old credentials never bleed across profiles
    legacy_token = DEFAULT_CLI_DIR / "token.json"
    if legacy_token.exists():
        try:
            legacy_token.unlink()
        except Exception:
            pass

    # 1. Restore token to keyring
    token_file = pdir / "token.json"
    token_str = ""
    if token_file.exists():
        try:
            token_str = token_file.read_text(encoding="utf-8").strip()
        except Exception:
            token_str = ""

    if is_valid_token_json(token_str):
        keyring_ok = False
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, token_str)
            keyring_ok = True
        except Exception as e:
            print(f"[Warning] Failed to write token to OS keyring: {e}", file=sys.stderr)
        # Write fallback file if keyring write failed
        if not keyring_ok:
            try:
                legacy_token.write_text(token_str, encoding="utf-8")
            except Exception as e:
                print(f"[Warning] Failed to write fallback token file: {e}", file=sys.stderr)
    else:
        # Empty profile or invalid credentials: clear both OS keyring AND legacy token
        clear_active_credentials()

    # 2. Restore sessions.json
    prof_sessions = pdir / "sessions.json"
    cli_sessions = DEFAULT_CLI_DIR / "sessions.json"
    if prof_sessions.exists():
        try:
            shutil.copy2(prof_sessions, cli_sessions)
        except Exception:
            pass
    elif cli_sessions.exists():
        # Remove active session file so we don't bleed sessions across accounts
        try:
            cli_sessions.unlink()
        except Exception:
            pass

    # 3. Restore settings.json
    prof_settings = pdir / "settings.json"
    cli_settings = DEFAULT_CLI_DIR / "settings.json"
    if prof_settings.exists():
        try:
            shutil.copy2(prof_settings, cli_settings)
        except Exception:
            pass

    set_active_profile_name(profile_name)


def cmd_status():
    ensure_dirs()
    curr = get_active_profile_name()

    print("==================================================")
    print("   Google Colab CLI Multi-Account Profile Status  ")
    print("==================================================")
    print(f"Active Profile : {curr}")

    curr_dir = get_profile_dir(curr)
    token_file = curr_dir / "token.json"
    token_str = token_file.read_text(encoding="utf-8").strip() if token_file.exists() else ""
    curr_email, curr_verified = get_profile_email_info(curr)

    if token_file.exists() and is_valid_token_json(token_str):
        tag = curr_email if curr_verified else f"{curr_email} (label, unverified)"
        email_str = f" ({tag})" if curr_email else ""
        print(f"Auth Status    : Authenticated{email_str}")
    else:
        tag = curr_email if curr_verified else f"{curr_email} (label, unverified)"
        email_str = f" ({tag})" if curr_email else ""
        print(f"Auth Status    : Not Authenticated{email_str} (will prompt login on next colab command)")

    sessions_file = DEFAULT_CLI_DIR / "sessions.json"
    session_count = 0
    if sessions_file.exists():
        try:
            data = json.loads(sessions_file.read_text(encoding="utf-8"))
            session_count = len(data)
        except Exception:
            pass
    print(f"Saved Sessions : {session_count}")
    print(f"Config Dir     : {curr_dir}")
    print("--------------------------------------------------")
    print("Available Profiles:")

    profiles = sorted([p.name for p in PROFILES_DIR.iterdir() if p.is_dir() and is_valid_profile_name(p.name)])
    if not profiles:
        profiles = ["account1", "account2"]
        for p in profiles:
            get_profile_dir(p)

    for p in profiles:
        mark = " [*] ACTIVE" if p == curr else " [ ]"
        p_dir = get_profile_dir(p)
        p_token = p_dir / "token.json"
        p_tok_str = p_token.read_text(encoding="utf-8").strip() if p_token.exists() else ""
        sess_file = p_dir / "sessions.json"
        p_sess_count = 0
        if sess_file.exists():
            try:
                p_sess_count = len(json.loads(sess_file.read_text(encoding="utf-8")))
            except Exception:
                pass

        p_email, p_verified = get_profile_email_info(p)
        if p_token.exists() and is_valid_token_json(p_tok_str):
            if p_email:
                tag = p_email if p_verified else f"{p_email} (label, unverified)"
                em_str = f" - Logged in as {tag}"
            else:
                em_str = " - Authenticated"
        else:
            if p_email:
                tag = p_email if p_verified else f"{p_email} (label, unverified)"
                em_str = f" ({tag}) - No credentials saved"
            else:
                em_str = " - No credentials saved"
        print(f"  {mark} {p:<12}{em_str} ({p_sess_count} sessions)")
    print("==================================================")


def cmd_switch(target_profile: str):
    target = target_profile.strip()
    if target in ["1"]:
        target = "account1"
    elif target in ["2"]:
        target = "account2"

    if not is_valid_profile_name(target):
        print(f"[Error] Invalid profile name '{target}'. Name must contain only alphanumeric characters, dashes, or underscores.", file=sys.stderr)
        sys.exit(1)

    ensure_dirs()
    curr = get_active_profile_name()
    if curr == target:
        load_profile_state(target)
        print(f"[Info] Profile '{target}' is already active.")
        cmd_status()
        return

    print(f"[Colab Switcher] Saving state for current profile '{curr}'...")
    save_current_state_to_profile(curr)

    print(f"[Colab Switcher] Switching to profile '{target}'...")
    load_profile_state(target)

    pdir = get_profile_dir(target)
    token_file = pdir / "token.json"
    token_str = token_file.read_text(encoding="utf-8").strip() if token_file.exists() else ""
    target_email, target_verified = get_profile_email_info(target)

    if token_file.exists() and is_valid_token_json(token_str):
        tag = target_email if target_verified else f"{target_email} (label, unverified)"
        email_str = f" ({tag})" if target_email else ""
        print(f"[Colab Switcher] Successfully switched to '{target}'{email_str}!")
    else:
        tag = target_email if target_verified else f"{target_email} (label, unverified)"
        email_str = f" ({tag})" if target_email else ""
        print(f"[Colab Switcher] Switched to '{target}'{email_str}.")
        print("[Note] This profile is not authenticated yet.")
        print("       Run 'colab ssh' or 'colab sessions' to initiate Google login for this account.")


def cmd_save(target_profile: Optional[str] = None):
    ensure_dirs()
    target = (target_profile or get_active_profile_name()).strip()
    if target in ["1"]:
        target = "account1"
    elif target in ["2"]:
        target = "account2"

    if not is_valid_profile_name(target):
        print(f"[Error] Invalid profile name: '{target}'", file=sys.stderr)
        sys.exit(1)

    saved = save_current_state_to_profile(target)
    if saved:
        print(f"[Colab Switcher] Snapshot of profile '{target}' successfully saved.")
    else:
        print(f"[Warning] No active OAuth credentials found in OS keyring or legacy token file for profile '{target}'.")
        print(f"[Colab Switcher] Saved session and configuration state for profile '{target}'.")


def cmd_logout(target_profile: Optional[str] = None):
    ensure_dirs()
    profile = (target_profile or get_active_profile_name()).strip()
    if profile in ["1"]:
        profile = "account1"
    elif profile in ["2"]:
        profile = "account2"

    if not is_valid_profile_name(profile):
        print(f"[Error] Invalid profile name: '{profile}'", file=sys.stderr)
        sys.exit(1)

    pdir = get_profile_dir(profile)
    token_file = pdir / "token.json"
    email_file = pdir / "email.txt"
    if token_file.exists():
        token_file.unlink()
    if email_file.exists():
        email_file.unlink()

    curr = get_active_profile_name()
    if profile == curr:
        clear_active_credentials()

    print(f"[Colab Switcher] Logged out credentials for profile '{profile}'.")


def cmd_set_email(profile: str, email: str):
    ensure_dirs()
    target = profile.strip()
    if target in ["1"]:
        target = "account1"
    elif target in ["2"]:
        target = "account2"

    if not is_valid_profile_name(target):
        print(f"[Error] Invalid profile name: '{target}'", file=sys.stderr)
        sys.exit(1)

    pdir = get_profile_dir(target)
    (pdir / "email.txt").write_text(email.strip(), encoding="utf-8")
    print(f"[Colab Switcher] Saved email label '{email.strip()}' for profile '{target}'.")


def main():
    if len(sys.argv) < 2:
        cmd_status()
        return

    action = sys.argv[1].lower()
    if action in ["status", "info"]:
        cmd_status()
    elif action in ["switch", "use"]:
        if len(sys.argv) < 3:
            print("Usage: python colab_switch.py switch <account1|account2|profile_name>")
            sys.exit(1)
        cmd_switch(sys.argv[2])
    elif action in ["save", "backup"]:
        target = sys.argv[2] if len(sys.argv) >= 3 else None
        cmd_save(target)
    elif action in ["logout", "clear"]:
        target = sys.argv[2] if len(sys.argv) >= 3 else None
        cmd_logout(target)
    elif action in ["set-email", "email", "label"]:
        if len(sys.argv) < 4:
            print("Usage: python colab_switch.py set-email <profile_name> <email_or_label>")
            sys.exit(1)
        cmd_set_email(sys.argv[2], sys.argv[3])
    elif action in ["list", "ls"]:
        ensure_dirs()
        profiles = sorted([p.name for p in PROFILES_DIR.iterdir() if p.is_dir() and is_valid_profile_name(p.name)])
        print("Profiles:", ", ".join(profiles))
    elif action in ["help", "-h", "--help", "/?"]:
        print("Usage: python colab_switch.py [status|switch <name>|save|logout [name]|set-email <name> <email>|list|help]")
    else:
        # Default to treating argument as target profile name (e.g. `colab_switch.py account2`)
        cmd_switch(action)


if __name__ == "__main__":
    main()
