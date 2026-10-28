from datetime import datetime
import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["smartmess"]
meals = db["meals"]

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "KNN.joblib")
try:
    ml_model = joblib.load(MODEL_PATH)
except Exception as e:
    ml_model = None
    print(f"Warning: Failed to load ML model from {MODEL_PATH}: {e}")

DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
MEAL_TYPES = ["Breakfast", "Lunch", "Dinner"]
WEEKEND_DAYS = {"Saturday", "Sunday"}


def predict_food_waste(features_data):
    if ml_model is None:
        raise ValueError("ML model is not available. Please verify model/KNN.joblib exists.")

    day_of_week = features_data.get("day_of_week")
    meal_type = features_data.get("meal_type")
    menu = str(features_data.get("menu", "")).strip()

    students_served = float(features_data.get("students_served", 0))
    food_prepared_kg = float(features_data.get("food_prepared_kg", 0))

    is_weekend = features_data.get("is_weekend")
    if is_weekend is None:
        is_weekend = 1 if day_of_week in WEEKEND_DAYS else 0
    else:
        is_weekend = 1 if str(is_weekend).lower() in ["1", "true"] else 0

    df = pd.DataFrame([{
        "day_of_week": day_of_week,
        "meal_type": meal_type,
        "menu": menu,
        "students_served": students_served,
        "food_prepared_kg": food_prepared_kg,
        "is_weekend": is_weekend
    }])

    pred = float(ml_model.predict(df)[0])
    predicted_waste_kg = round(max(0.0, pred), 2)
    waste_percent = (
        round(predicted_waste_kg / food_prepared_kg * 100, 2)
        if food_prepared_kg > 0
        else 0.0
    )
    predicted_consumed_kg = round(
        max(0.0, food_prepared_kg - predicted_waste_kg), 2
    )

    return {
        "predicted_food_wasted_kg": predicted_waste_kg,
        "predicted_waste_percent": waste_percent,
        "predicted_food_consumed_kg": predicted_consumed_kg
    }


@app.route("/")
def home():
    return jsonify({
        "message": "SmartMess API is running"
    })


@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/api/summary")
def summary():
    records = list(meals.find({}, {"_id": 0}))

    def total(field):
        return round(sum(record.get(field, 0) for record in records), 2)

    food_prepared_kg = total("food_prepared_kg")
    food_consumed_kg = total("food_consumed_kg")
    food_wasted_kg = total("food_wasted_kg")
    students_served = total("students_served")

    waste_percent = (
        round(food_wasted_kg / food_prepared_kg * 100, 2)
        if food_prepared_kg > 0
        else 0
    )

    return jsonify({
        "expected_students": 742,
        "students_served": students_served,
        "food_prepared_kg": food_prepared_kg,
        "food_consumed_kg": food_consumed_kg,
        "food_wasted_kg": food_wasted_kg,
        "waste_percent": waste_percent
    })


@app.route("/api/db-test")
def db_test():
    try:
        client.admin.command("ping")
        return jsonify({
            "message": "MongoDB connected successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    day_of_week = data.get("day_of_week")
    meal_type = data.get("meal_type")
    menu = str(data.get("menu", "")).strip()

    if day_of_week not in DAYS:
        return jsonify({"error": "Please choose a valid day of the week"}), 400

    if meal_type not in MEAL_TYPES:
        return jsonify({"error": "Please choose a valid meal type"}), 400

    if not menu:
        return jsonify({"error": "Menu is required"}), 400

    try:
        students_served = float(data.get("students_served"))
        food_prepared_kg = float(data.get("food_prepared_kg"))
    except (TypeError, ValueError):
        return jsonify({
            "error": "Students served and food prepared must be numbers"
        }), 400

    if min(students_served, food_prepared_kg) < 0:
        return jsonify({"error": "Values cannot be negative"}), 400

    try:
        result = predict_food_waste(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/meals", methods=["POST"])
def add_meal():
    data = request.get_json(silent=True) or {}

    day_of_week = data.get("day_of_week")
    meal_type = data.get("meal_type")
    menu = str(data.get("menu", "")).strip()

    if day_of_week not in DAYS:
        return jsonify({"error": "Please choose a valid day of the week"}), 400

    if meal_type not in MEAL_TYPES:
        return jsonify({"error": "Please choose a valid meal type"}), 400

    if not menu:
        return jsonify({"error": "Menu is required"}), 400

    try:
        students_served = int(data.get("students_served"))
        food_prepared_kg = float(data.get("food_prepared_kg"))
    except (TypeError, ValueError):
        return jsonify({
            "error": "Students served and food prepared must be numbers"
        }), 400

    raw_consumed = data.get("food_consumed_kg")
    if raw_consumed is not None and str(raw_consumed).strip() != "":
        try:
            food_consumed_kg = float(raw_consumed)
            if food_consumed_kg < 0:
                return jsonify({"error": "Food consumed cannot be negative"}), 400
            if food_consumed_kg > food_prepared_kg:
                return jsonify({
                    "error": "Food consumed cannot exceed food prepared"
                }), 400
        except (TypeError, ValueError):
            return jsonify({"error": "Food consumed must be a number"}), 400
    else:
        food_consumed_kg = None

    if min(students_served, food_prepared_kg) < 0:
        return jsonify({"error": "Values cannot be negative"}), 400

    is_weekend = 1 if day_of_week in WEEKEND_DAYS else 0

    predicted_food_wasted_kg = None
    if ml_model is not None:
        try:
            pred_res = predict_food_waste({
                "day_of_week": day_of_week,
                "meal_type": meal_type,
                "menu": menu,
                "students_served": students_served,
                "food_prepared_kg": food_prepared_kg,
                "is_weekend": is_weekend
            })
            predicted_food_wasted_kg = pred_res["predicted_food_wasted_kg"]
            if food_consumed_kg is None:
                food_consumed_kg = pred_res["predicted_food_consumed_kg"]
        except Exception:
            pass

    # If consumed was not given and model prediction was unavailable, default to 0
    if food_consumed_kg is None:
        food_consumed_kg = 0.0

    # Calculated on the server so the stored data is always consistent
    food_wasted_kg = round(food_prepared_kg - food_consumed_kg, 2)
    waste_percent = (
        round(food_wasted_kg / food_prepared_kg * 100, 2)
        if food_prepared_kg > 0
        else 0.0
    )

    meal = {
        "day_of_week": day_of_week,
        "meal_type": meal_type,
        "menu": menu,
        "students_served": students_served,
        "food_prepared_kg": round(food_prepared_kg, 2),
        "food_consumed_kg": round(food_consumed_kg, 2),
        "food_wasted_kg": food_wasted_kg,
        "waste_percent": waste_percent,
        "is_weekend": is_weekend,
        "predicted_food_wasted_kg": predicted_food_wasted_kg,
        "created_at": datetime.now()
    }

    result = meals.insert_one(meal)

    return jsonify({
        "message": "Meal record saved",
        "id": str(result.inserted_id)
    }), 201


@app.route("/api/meals", methods=["GET"])
def get_meals():
    records = list(
        meals.find().sort("created_at", -1).limit(20)
    )

    for record in records:
        record["_id"] = str(record["_id"])

    return jsonify(records)


@app.route("/api/db-info")
def db_info():
    try:
        client.admin.command("ping")

        return jsonify({
            "database": db.name,
            "collection": meals.name,
            "connected_server": str(client.address),
            "connection_type": str(
                client.topology_description.topology_type_name
            )
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))