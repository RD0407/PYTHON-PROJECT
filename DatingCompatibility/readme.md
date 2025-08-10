# 💘 Compatibility Matcher

A detailed dating compatibility matcher in Python that compares your preferences against a list of candidates from a candidates.json file.
It calculates a compatibility percentage (0–100%) based on:

    🧠 Personality traits (20+ traits with ratings 0–10)

    💪 Physical preferences (gender-specific, with top 3 ranked per trait)

    📅 Age range (choose 2 ranges in 3-year increments)

    🌍 Nationality (continent-based, with “No Preference” option)

    🗣️ Languages (pick up to 3, with “No Preference” option)

    🚬 Smoking preference (Yes, No, or Doesn’t matter)

It also provides a summary chart for each candidate, showing:

    ✅ What matched

    ❌ What’s lacking

## 📦 Requirements

    Python 3.8+

    rich for fancy terminal UI

## Install dependencies:

pip install rich

## 📂 Project Structure

compatibility_matcher.py   # Main script
candidates.json            # Candidate data
README.md                  # This file

## 📝 Creating the candidates.json file

The JSON file should contain two main keys: "male" and "female".
Each contains a list of candidates with their attributes.

## Example:

{
  "female": [
    {
      "name": "Alice",
      "age": 25,
      "nationality": "Europe",
      "languages": ["English", "French"],
      "smoking": "No",
      "personality": {
        "Kindness": 9,
        "Humor": 7,
        "Intellect": 8
      },
      "physical": {
        "Height": "160–165",
        "Weight": "50–55",
        "Hair Color": "Brown",
        "Breast Size": "Medium"
      }
    }
  ],
  "male": [
    {
      "name": "Bob",
      "age": 30,
      "nationality": "North America",
      "languages": ["English", "German"],
      "smoking": "Yes",
      "personality": {
        "Kindness": 8,
        "Confidence": 7,
        "Humor": 6
      },
      "physical": {
        "Height": "180–185",
        "Weight": "70–80",
        "Hair Color": "Black",
        "Facial Hair": "Short beard"
      }
    }
  ]
}

## ▶️ Running the Script

python compatibility_matcher.py

## 🖥️ How It Works

    Choose your partner’s gender (male or female).

    Set preferences:

        Age range (two choices)

        Nationality (continent)

        Languages (up to 3)

        Smoking preference (Yes/No/Doesn’t matter)

    Select personality traits:

        Pick top 6 traits

        Rate each on a scale of 0–10

    Select physical traits:

        For each, choose up to 3 preferences (0 = No preference)

    The script scores all candidates and displays:

        Compatibility %

        Summary: Why they match & what’s lacking

## 📊 Scoring System

    Personality → 40% weight

    Physical → 50% weight

    Misc (age, nationality, language, smoking) → 10% weight

## 💡 Notes

    Pressing Enter without a value uses default options (no crash).

    Choosing 0 (No Preference) means that category doesn’t affect the score.

    To make results more meaningful, ensure candidates.json has detailed, realistic data.
