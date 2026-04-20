import random
import uuid

ADJECTIVES = [

    # PURETÉ / SACRÉ
    "Sanctified", "Consecrated", "Purified", "Hallowed",
    "Sacrosanct", "Divine", "Blessed", "Exorcised", "Sacred",

    # PSYKER / ESPRIT
    "Psychic", "Empyrean", "Warpbound", "Mindforged",
    "Soulbound", "Astral", "Telepathic", "Psyonic",

    # GUERRE / DISCIPLINE
    "Astartes", "Titanic", "Ironclad", "Unyielding",
    "Relentless", "Vigilant", "Unbreakable", "Indomitus",

    # DARK / POWER
    "Black", "Obsidian", "Crimson", "Scarlet", "Nocturne",
    "Eclipsed", "Void", "Shadow", "Umbral", "Abyssal",
    "Infernal", "Phantom", "Spectral", "Dread", "Cursed",

    # PRESTIGE / NOBILITY
    "Sovereign", "Imperial", "Royal", "Aurelian", "Regal",
    "Noble", "Grand", "Supreme", "Majestic", "Exalted",
    "Eternal", "Prime", "Absolute",

    # MYSTIC / ARCANE
    "Arcane", "Celestial", "Ethereal", "Runic",
    "Occult", "Mythic", "Enigmatic", "Lucent", "Radiant",
    "Transcendent", "Timeless", "Forgotten", "Ancient",

    # TECH / AI
    "Quantum", "Neural", "Synthetic", "Cyber", "Digital",
    "Algorithmic", "Fractal", "Singular", "Adaptive", "Autonomous",

    # AESTHETIC / STYLE
    "Velvet", "Ivory", "Golden", "Silver", "Frozen",
    "Crystal", "Polished", "Refined", "Luminous", "Chromatic",

    # NORDIC / ENTITÉS
    "Odinic", "Thoric", "Loki", "Freyr", "Tyr", "Heimdall",
    "Baldr", "Njord", "Vidar", "Forseti",

    # COSMOS / DESTIN
    "Asgardian", "Midgardian", "Jotunn", "Vanir", "Aesir",
    "Ragnarok", "Fated", "Doomed", "Runebound",

    # ENVIRONNEMENT
    "Frost", "Glacial", "Icy", "Stormborn", "Thunder",
    "Blizzard", "Winterborn", "Iceforged", "Snowbound",

    # GUERRIER
    "Berserker", "Drengr", "Shielded", "Ironbound",
    "Warborn", "Bloodbound", "Battleforged", "Skaldic",

    # MYSTIQUE NORDIQUE
    "Seidr", "Wyrd", "Eldritch", "Mythborn",
    "Grim", "Shadowborn", "Nightforged"
]

NOUNS = [

    # OBJECTS / SYMBOLS
    "Pearl", "Crown", "Throne", "Sigil", "Relic",
    "Artifact", "Emblem", "Seal", "Shard", "Core",

    # POWER STRUCTURES
    "Dominion", "Empire", "Order", "Legion", "Sanctum",
    "Citadel", "Stronghold", "Consortium", "Authority",

    # TECH / SYSTEMS
    "Engine", "Protocol", "Matrix", "System", "Framework",
    "Reactor", "Kernel", "Network", "Interface", "Module",

    # MYSTIC / ABSTRACT
    "Oracle", "Nexus", "Monolith", "Paradox", "Eclipse",
    "Continuum", "Singularity", "Axiom", "Vector", "Horizon",

    # DEFENSE / PRESENCE
    "Sentinel", "Guardian", "Warden", "Vanguard", "Overseer",

    # CONTAINMENT / VALUE
    "Vault", "Archive", "Repository", "Chamber", "Sanctuary",

    # NORDIC FIGURES
    "Odin", "Thor", "Loki", "Fenrir", "Jormungandr",
    "Ymir", "Baldr", "Freya", "Hel", "Skadi",

    # COSMOS
    "Asgard", "Midgard", "Niflheim", "Muspelheim",
    "Valhalla", "Fimbulwinter", "Bifrost", "Yggdrasil",

    # RELICS
    "Mjolnir", "Gungnir", "Draupnir", "Skidbladnir",
    "Runestone", "Totem",

    # WAR
    "Valkyrie", "Einherjar", "Longship", "Warband",
    "Shieldwall",

    # ABSTRACT
    "Wyrd", "Saga", "Abyss", "Fate"
]

def generate_unique_model_name():
    base_name = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
    unique_suffix = uuid.uuid4().hex[:8]
    return f"{base_name}__{unique_suffix}"


if __name__ == "__main__":
    for _ in range(5):
        print(generate_unique_model_name())
