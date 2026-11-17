#!/usr/bin/env python3
"""
fork_conversation.py

Clones/forks an Antigravity conversation by its Conversation ID into a new,
independent session that appears in the IDE chat history.

Usage:
    python fork_conversation.py <source_conversation_id> [--title "Custom Fork Title"]
    python fork_conversation.py --list

Examples:
    # 1. List all recent conversations to find IDs:
    python fork_conversation.py --list

    # 2. Fork a conversation by ID (defaults to original title + ' (Fork)'):
    python fork_conversation.py fb4c85b4-0b3d-4d34-99bd-24a454b20862

    # 3. Fork a conversation with a custom title:
    python fork_conversation.py fb4c85b4-0b3d-4d34-99bd-24a454b20862 --title "Diet-Microbiome: Neural gLV Exploration Branch"
"""

import sys
import os
import uuid
import sqlite3
import shutil
import argparse
from datetime import datetime, timezone


def get_antigravity_dir() -> str:
    user_home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not user_home:
        raise RuntimeError("Could not determine user home directory.")
    base = os.path.join(user_home, ".gemini", "antigravity")
    if not os.path.exists(base):
        raise FileNotFoundError(f"Antigravity directory not found at: {base}")
    return base


def list_conversations(base_dir: str):
    summ_path = os.path.join(base_dir, "conversation_summaries.db")
    if not os.path.exists(summ_path):
        print(f"No conversation_summaries.db found at: {summ_path}")
        return

    con = sqlite3.connect(summ_path)
    cur = con.cursor()
    cur.execute("SELECT conversation_id, title, last_modified_time FROM conversation_summaries ORDER BY last_modified_time DESC LIMIT 25")
    rows = cur.fetchall()
    con.close()

    print(f"\nRecent Conversations ({len(rows)} found):")
    print(f"{'Conversation ID':<38} | {'Last Modified':<26} | {'Title'}")
    print("-" * 100)
    for cid, title, last_mod in rows:
        title_str = (title[:40] + "...") if title and len(title) > 40 else (title or "Untitled")
        print(f"{cid:<38} | {str(last_mod):<26} | {title_str}")
    print()


def fork_conversation(source_id: str, custom_title: str | None = None) -> str:
    base_dir = get_antigravity_dir()
    new_id = str(uuid.uuid4())

    src_db = os.path.join(base_dir, "conversations", f"{source_id}.db")
    if not os.path.exists(src_db):
        raise FileNotFoundError(f"Source conversation database not found: {src_db}")

    dst_db = os.path.join(base_dir, "conversations", f"{new_id}.db")

    print(f"[*] Forking conversation:")
    print(f"    Source ID: {source_id}")
    print(f"    New ID:    {new_id}")

    # 1. Safely clone SQLite DB via backup API (captures uncheckpointed WAL pages)
    print("[*] Duplicating conversation database...")
    con_src = sqlite3.connect(src_db)
    con_dst = sqlite3.connect(dst_db)
    con_src.backup(con_dst)
    con_src.close()

    # Update cascade_id references in trajectory_meta
    cur_dst = con_dst.cursor()
    try:
        cur_dst.execute("UPDATE trajectory_meta SET cascade_id = ?", (new_id,))
        con_dst.commit()
    except Exception as e:
        print(f"    [!] Note on trajectory_meta: {e}")
    con_dst.close()

    # 2. Duplicate brain directory if present
    src_brain = os.path.join(base_dir, "brain", source_id)
    dst_brain = os.path.join(base_dir, "brain", new_id)
    if os.path.exists(src_brain):
        print("[*] Duplicating brain directory...")
        shutil.copytree(src_brain, dst_brain)

    # 3. Duplicate annotations file if present
    src_anno = os.path.join(base_dir, "annotations", f"{source_id}.pbtxt")
    dst_anno = os.path.join(base_dir, "annotations", f"{new_id}.pbtxt")
    if os.path.exists(src_anno):
        print("[*] Duplicating annotations...")
        with open(src_anno, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(source_id, new_id)
        with open(dst_anno, "w", encoding="utf-8") as f:
            f.write(content)

    # 4. Insert row into conversation_summaries.db so it appears in IDE history
    summ_path = os.path.join(base_dir, "conversation_summaries.db")
    if os.path.exists(summ_path):
        print("[*] Registering fork in conversation_summaries.db...")
        con_summ = sqlite3.connect(summ_path)
        cur_summ = con_summ.cursor()
        cur_summ.execute("SELECT * FROM conversation_summaries WHERE conversation_id = ?", (source_id,))
        row = cur_summ.fetchone()

        if row:
            row_list = list(row)
            original_title = row_list[1] or "Conversation"
            fork_title = custom_title or f"{original_title} (Fork)"

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

            row_list[0] = new_id                 # conversation_id
            row_list[1] = fork_title             # title
            row_list[2] = fork_title             # summary
            row_list[4] = now_iso                # updated_at
            row_list[6] = "CASCADE_RUN_STATUS_IDLE" # ready state
            row_list[16] = now_iso               # client_updated_at

            # Replace raw conversation ID inside binary protobuf blob if present
            if len(row_list) > 19 and isinstance(row_list[19], bytes):
                row_list[19] = row_list[19].replace(source_id.encode("utf-8"), new_id.encode("utf-8"))

            placeholders = ", ".join(["?"] * len(row_list))
            cur_summ.execute(f"INSERT INTO conversation_summaries VALUES ({placeholders})", row_list)
            con_summ.commit()
        con_summ.close()

    print("\n[+] Fork completed successfully!")
    print(f"    New Conversation ID: {new_id}")
    print("    You can now find and click this branch in your IDE chat history panel.\n")
    return new_id


def main():
    parser = argparse.ArgumentParser(description="Fork an Antigravity conversation by ID.")
    parser.add_argument("conversation_id", nargs="?", help="Source conversation UUID to fork")
    parser.add_argument("--title", help="Custom title for the new forked conversation")
    parser.add_argument("--list", action="store_true", help="List recent conversation IDs")

    args = parser.parse_args()
    base_dir = get_antigravity_dir()

    if args.list:
        list_conversations(base_dir)
        return

    if not args.conversation_id:
        parser.print_help()
        print("\nTip: Use --list to view recent conversation IDs.")
        sys.exit(1)

    fork_conversation(args.conversation_id, custom_title=args.title)


if __name__ == "__main__":
    main()
