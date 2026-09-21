#!/usr/bin/env python3
"""Check the shipped iOS 27 App Intents registration, without launching the app."""
import argparse
import json
import plistlib
from pathlib import Path
import sys


def verify(app: Path) -> None:
    metadata = app / "Metadata.appintents" / "extract.actionsdata"
    data = json.loads(metadata.read_text())
    actions = data["actions"]
    # These are Xcode 27 RC's serialized policy values. Fail visibly if a
    # future metadata format changes; do not silently skip authentication.
    required = {
        "AskOfflineQuestionIntent": (1, False),
        "ContinueQuestionInZimfoIntent": (1, True),
        "SearchOfflineContentIntent": (2, True),
        "NearbyHereIntent": (1, False),
        "NearbyPlaceIntent": (1, False),
    }
    for name, (policy, foreground) in required.items():
        action = actions[name]
        if action["authenticationPolicy"] != policy:
            raise ValueError(f"{name}: unexpected authentication policy")
        if action["openAppWhenRun"] != foreground:
            raise ValueError(f"{name}: unexpected foreground behavior")
    schemas = actions["SearchOfflineContentIntent"]["assistantDefinedSchemas"]
    if not any(s["domain"] == "system" and s["name"] == "SystemSearchInAppIntent" for s in schemas):
        raise ValueError("Missing iOS 27 search-in-app schema")
    if "ZimfoArticleEntity" not in data["entities"]:
        raise ValueError("Missing offline article entity")
    if "ZimfoCurrentArticleEntity" not in data["entities"]:
        raise ValueError("Missing current-document entity")
    current = data["entities"]["ZimfoCurrentArticleEntity"]
    transfer = current["transferableContentTypes"]
    if {item["contentType"] for item in transfer["exportableTypes"]} != {"public.plain-text"} or transfer["importableTypes"]:
        raise ValueError("Current document must offer export-only plain text")
    if current["assistantDefinedSchemas"]:
        raise ValueError("Current document must not impersonate a domain schema")
    info = plistlib.loads((app / "Info.plist").read_bytes())
    if "com.tiltastech.zimfo.current-article" not in info.get("NSUserActivityTypes", []):
        raise ValueError("Missing current-document activity registration")
    shortcuts = {s["actionIdentifier"]: s for s in data["autoShortcuts"]}
    questions = shortcuts["AskOfflineQuestionIntent"]
    phrases = {p["key"] for p in questions["phraseTemplates"]}
    if "Ask an offline question with ${applicationName}" not in phrases:
        raise ValueError("Missing explicit offline-question trigger")
    if "Ask ${applicationName}" in phrases:
        raise ValueError("Ambiguous bare Ask-app trigger is still registered")
    for name, shortcut in shortcuts.items():
        if name == "AskOfflineQuestionIntent":
            continue
        if phrases.intersection(p["key"] for p in shortcut["phraseTemplates"]):
            raise ValueError(f"Question phrase also points to {name}")
    parameters = {p["name"]: p for p in actions["AskOfflineQuestionIntent"]["parameters"]}
    if parameters["question"]["isOptional"] is not True:
        raise ValueError("Question must be prompted inside perform() so its lifecycle is logged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path, help="Signed iPhone .app built with Xcode 27")
    args = parser.parse_args()
    try:
        verify(args.app)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Siri metadata gate: FAILED: {error}", file=sys.stderr)
        sys.exit(1)
    print("Siri metadata gate: OK (actions, authentication, foreground behavior, entity, search schema)")
