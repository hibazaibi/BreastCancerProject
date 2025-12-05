# 🩺 BreastCare AI  
Plateforme intelligente pour l’analyse des mammographies, échographies et l’accompagnement des patientes via un assistant vocal multilingue.

## 🚀 Introduction
BreastCare AI est une plateforme web complète qui utilise l’intelligence artificielle pour assister les professionnels de santé dans la détection précoce du cancer du sein.  
Elle combine :

- Analyse automatique des mammographies (EfficientNet)
- Analyse des échographies (ResNet50V2)
- Assistant vocal intelligent multilingue (FR/EN/AR)
- Recommandations de traitements personnalisées
- Génération d’un rapport PDF unifié
- Dashboard administrateur avec statistiques

Ce projet vise à améliorer la rapidité, la précision et l’accessibilité du dépistage.

---

## 🧠 Fonctionnalités principales

### 🔍 Analyse d’images médicales
- Upload des images mammographiques ou échographiques  
- Classification : Bénin / Malin (+ Normal pour échographie)  
- Score de confiance et prédictions IA  
- Stockage sécurisé dans PostgreSQL  

### 🗣 Assistant vocal (Groq Whisper + LLaMA)
- Détection automatique de la langue  
- Analyse des symptômes exprimés oralement  
- Réponses empathiques et adaptées  
- Évaluation automatique de la sévérité (urgent, élevé, modéré, faible)

### 📄 Rapport PDF
- Informations patient  
- Images analysées  
- Résultats + probabilités  
- Recommandation thérapeutique  
- Téléchargement direct depuis l’interface

### 📊 Dashboard Administrateur
- Nombre total de patientes  
- Diagnostics mammographie / échographie  
- Graphiques : bénin/malin, traitements recommandés, évolution par date  
- Historique complet par patiente  

---

## 🏗 Architecture du système
Frontend → HTML / CSS / JS / Chart.js
Backend → Flask, Python, TensorFlow, Keras
IA → EfficientNetB0 (mammo), ResNet50V2 (écho), Whisper, LLaMA
Database → PostgreSQL
