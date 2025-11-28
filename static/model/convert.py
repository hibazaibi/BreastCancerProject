# convertir.py  ← Colle tout ça dedans
import os
import subprocess
import sys

print("=" * 70)
print("CONVERSION KERAS → TENSORFLOW.JS")
print("=" * 70)

# Chemin exact vers ton modèle (à changer si besoin)
keras_file = "static/model/breast_cancer_model.keras"

if not os.path.exists(keras_file):
    print(f"Fichier introuvable : {keras_file}")
    print("Assure-toi que le chemin est correct !")
    input("Appuie sur Entrée pour quitter...")
    sys.exit(1)

print(f"Fichier trouvé : {keras_file}")

output_dir = "static/model"  # On écrase directement au bon endroit

print(f"Conversion vers → {output_dir}")

cmd = [
    "tensorflowjs_converter",
    "--input_format=keras",
    keras_file,
    output_dir
]

try:
    print("Conversion en cours... (30 sec à 2 min)")
    subprocess.run(cmd, check=True)
    print("CONVERSION RÉUSSIE !")

    print("\nFichiers générés :")
    for f in os.listdir(output_dir):
        if f.endswith('.bin') or f == 'model.json':
            size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
            print(f"   • {f} ({size:.2f} MB)")

    print("\nTOUT EST PRÊT !")
    print("Tu peux maintenant lancer ton app Flask")
    print("Le modèle se chargera sans erreur 404")

except subprocess.CalledProcessError as e:
    print("ERREUR DE CONVERSION")
    print("Solution rapide : ouvre un terminal et tape ça →")
    print(f"pip install tensorflow==2.16.1")
    print(f"pip install \"tensorflowjs==4.21.0\" --no-deps")
    print(f"puis relance ce script")

input("\nAppuie sur Entrée pour fermer...")