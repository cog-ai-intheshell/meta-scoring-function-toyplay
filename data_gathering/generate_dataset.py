import argparse
import csv
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config


def momentum_from_number_label(number_label, min_label):
    """Convertit un label de chiffre en cible binaire selon le seuil configure."""
    return 1 if number_label >= int(min_label) else 0


def read_digits_rows(path, min_label):
    """Charge le CSV source des digits et reconstruit la cible binaire momentum."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Aucune ligne trouvee dans {path}")

    pixel_cols = [name for name in rows[0].keys() if name.startswith("pixel_")]
    if not pixel_cols:
        raise ValueError(f"Aucune colonne pixel_ trouvee dans {path}")

    if "number_label" not in rows[0]:
        raise ValueError(f"Colonne 'number_label' absente dans {path}")

    normalized_rows = []
    for original_index, row in enumerate(rows):
        number_label = int(float(row["number_label"]))
        momentum = momentum_from_number_label(number_label, min_label)
        normalized = {"original_index": original_index, "momentum": momentum}
        for col in pixel_cols:
            normalized[col] = float(row[col])
        normalized_rows.append(normalized)

    return normalized_rows, pixel_cols


def read_momentum_template(path):
    """Charge la sequence binaire de reference utilisee comme gabarit de regimes."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "momentum" not in reader.fieldnames:
            raise ValueError(f"Colonne 'momentum' absente dans {path}")
        return [int(float(row["momentum"])) for row in reader]


def extract_runs(values):
    """Decompose une suite binaire en runs consecutifs valeur-longueur."""
    if not values:
        return []

    runs = []
    current_value = int(values[0])
    current_length = 1

    for value in values[1:]:
        value = int(value)
        if value == current_value:
            current_length += 1
            continue

        runs.append((current_value, current_length))
        current_value = value
        current_length = 1

    runs.append((current_value, current_length))
    return runs


def allocate_run_lengths(template_lengths, target_total):
    """Repartit une somme cible sur des runs en conservant leur structure relative."""
    n_runs = len(template_lengths)
    if n_runs == 0:
        if target_total != 0:
            raise ValueError("Impossible d'allouer une somme positive sans run template")
        return []

    if target_total < n_runs:
        raise ValueError(
            f"Somme cible trop petite ({target_total}) pour garantir au moins 1 element sur {n_runs} runs."
        )

    base = [1] * n_runs
    remaining = target_total - n_runs

    if remaining == 0:
        return base

    total_weight = float(sum(template_lengths))
    raw_extras = [remaining * (length / total_weight) for length in template_lengths]
    extras = [int(value) for value in raw_extras]
    leftovers = remaining - sum(extras)

    ranked_indices = sorted(
        range(n_runs),
        key=lambda idx: (raw_extras[idx] - extras[idx], template_lengths[idx], -idx),
        reverse=True,
    )
    for idx in ranked_indices[:leftovers]:
        extras[idx] += 1

    return [b + e for b, e in zip(base, extras)]


def build_melania_style_sequence(template_values, n_zeros, n_ones):
    """Construit une sequence binaire de taille cible en reprenant le style des runs du template."""
    runs = extract_runs(template_values)
    zero_template_lengths = [length for value, length in runs if value == 0]
    one_template_lengths = [length for value, length in runs if value == 1]

    zero_lengths = allocate_run_lengths(zero_template_lengths, n_zeros)
    one_lengths = allocate_run_lengths(one_template_lengths, n_ones)

    zero_cursor = 0
    one_cursor = 0
    sequence = []

    for value, _ in runs:
        if value == 0:
            length = zero_lengths[zero_cursor]
            zero_cursor += 1
        else:
            length = one_lengths[one_cursor]
            one_cursor += 1

        sequence.extend([value] * length)

    return sequence


def write_dataset_csv(rows, pixel_cols, output_path):
    """Ecrit le dataset final reordonne au format CSV attendu par le pipeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataset_index", "original_index", *pixel_cols, "momentum"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_index, row in enumerate(rows):
            payload = {
                "dataset_index": dataset_index,
                "original_index": row["original_index"],
                "momentum": row["momentum"],
            }
            for col in pixel_cols:
                payload[col] = row[col]
            writer.writerow(payload)


def split_protocol_rows(rows, dev_windows, initial_train_size, window_size):
    """Decoupe la sequence temporelle en CSV dev et holdout sans melanger l'ordre."""
    split_index = int(initial_train_size) + int(dev_windows) * int(window_size)

    if split_index <= 0 or split_index >= len(rows):
        raise ValueError(
            "Le split dev/holdout est invalide pour la taille du dataset genere."
        )

    return rows[:split_index], rows[split_index:]


def summarize_sequence(values):
    """Resume une sequence binaire par sa taille, ses comptes et son nombre de switches."""
    switches = sum(left != right for left, right in zip(values, values[1:]))
    return {
        "n_samples": len(values),
        "zeros": int(sum(value == 0 for value in values)),
        "ones": int(sum(value == 1 for value in values)),
        "switches": int(switches),
    }


def main():
    """Point d'entree CLI pour generer le dataset final a partir des deux sources CSV."""
    parser = argparse.ArgumentParser(description="Genere un dataset digits reordonne dans le style des runs MELANIA.")
    parser.add_argument(
        "--digits-csv",
        default=str(config.GENERATOR_DIGITS_SOURCE_PATH),
        help=(
            "CSV source des digits avec colonnes pixel_* et number_label. "
            f"Defaut: {config.GENERATOR_DIGITS_SOURCE_PATH}"
        ),
    )
    parser.add_argument(
        "--template-csv",
        default=str(config.GENERATOR_REFERENCE_PATH),
        help=(
            "CSV template contenant une colonne momentum. "
            f"Defaut: {config.GENERATOR_REFERENCE_PATH}"
        ),
    )
    parser.add_argument("--output", default=str(config.GENERATOR_OUTPUT_PATH), help="Chemin de sortie du CSV final.")
    parser.add_argument(
        "--output-dev",
        default=str(config.GENERATOR_DEV_OUTPUT_PATH),
        help="Chemin de sortie du CSV dev (train initial + 120 premieres fenetres).",
    )
    parser.add_argument(
        "--output-holdout",
        default=str(config.GENERATOR_HOLDOUT_OUTPUT_PATH),
        help="Chemin de sortie du CSV holdout (50 dernieres fenetres).",
    )
    parser.add_argument("--random-state", type=int, default=config.GENERATOR_RANDOM_STATE, help="Seed pour melanger les pools positifs/negatifs.")
    parser.add_argument(
        "--digit-positive-min-label",
        type=int,
        default=config.DIGIT_POSITIVE_MIN_LABEL,
        help=(
            "Momentum vaut 1 si number_label >= cette valeur. "
            "Exemple: 5 donne 5..9, 7 donne 7..9."
        ),
    )
    args = parser.parse_args()

    digits_rows, pixel_cols = read_digits_rows(
        args.digits_csv,
        min_label=args.digit_positive_min_label,
    )
    template_values = read_momentum_template(args.template_csv)

    zero_rows = [dict(row) for row in digits_rows if row["momentum"] == 0]
    one_rows = [dict(row) for row in digits_rows if row["momentum"] == 1]

    target_sequence = build_melania_style_sequence(
        template_values,
        n_zeros=len(zero_rows),
        n_ones=len(one_rows),
    )

    if len(target_sequence) != len(digits_rows):
        raise ValueError("La sequence cible n'a pas la meme taille que le dataset digits.")

    rng = random.Random(args.random_state)
    rng.shuffle(zero_rows)
    rng.shuffle(one_rows)

    zero_cursor = 0
    one_cursor = 0
    reordered_rows = []

    for target_value in target_sequence:
        if target_value == 0:
            row = zero_rows[zero_cursor]
            zero_cursor += 1
        else:
            row = one_rows[one_cursor]
            one_cursor += 1

        row["momentum"] = target_value
        reordered_rows.append(row)

    write_dataset_csv(reordered_rows, pixel_cols, args.output)
    dev_rows, holdout_rows = split_protocol_rows(
        reordered_rows,
        dev_windows=config.N_WINDOWS,
        initial_train_size=config.INITIAL_TRAIN_SIZE,
        window_size=config.MODEL_LIFE_WINDOW,
    )
    write_dataset_csv(dev_rows, pixel_cols, args.output_dev)
    write_dataset_csv(holdout_rows, pixel_cols, args.output_holdout)

    summary = summarize_sequence(target_sequence)
    print(f"Saved to: {Path(args.output).resolve()}")
    print(f"Saved dev split to: {Path(args.output_dev).resolve()}")
    print(f"Saved holdout split to: {Path(args.output_holdout).resolve()}")
    print(f"n_samples: {summary['n_samples']}")
    print(f"n_samples_dev: {len(dev_rows)}")
    print(f"n_samples_holdout: {len(holdout_rows)}")
    print(f"zeros: {summary['zeros']}")
    print(f"ones: {summary['ones']}")
    print(f"switches: {summary['switches']}")
    print(f"digit_positive_min_label: {args.digit_positive_min_label}")


if __name__ == "__main__":
    main()
