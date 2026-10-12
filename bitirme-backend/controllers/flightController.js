import axios from "axios";

const MODEL_API_URL = process.env.MODEL_API_URL || "http://localhost:5050";

const getOneWay = async (req, res) => {
  const { from, to, date, adults, children, infants, cabinClass } = req.query;
  const options = {
    method: "GET",
    url: "https://flights-sky.p.rapidapi.com/flights/search-one-way",
    params: {
      fromEntityId: from,
      toEntityId: to,
      departDate: date,
      children: children,
      infants: infants,
      cabinClass: cabinClass,
      adults: adults,
    },
    headers: {
      "x-rapidapi-key": "3903953dacmsh58357f92fd8eea2p14c8e9jsn7e1dcb0762ff",
      "x-rapidapi-host": "flights-sky.p.rapidapi.com",
    },
  };

  try {
    const response = await axios.request(options);
    return res.status(200).json({ data: response.data });
  } catch (error) {
    console.error(error);
    return res.status(400).json({ message: "getOneWay not found." });
  }
};

const getOneWay7DaysMinPrice = async (req, res) => {
  const { fromId, toId, departDate, cabinClass } = req.query;
  const options = {
    method: "GET",
    url: "https://booking-com15.p.rapidapi.com/api/v1/flights/getMinPrice",
    params: {
      fromId: fromId + '.AIRPORT',
      toId: toId + '.AIRPORT',
      departDate: departDate,
      cabinClass: cabinClass,
      currency_code: 'USD'
    },
    headers: {
      'x-rapidapi-key': 'c2839ce5camsh1b729118ea46005p148515jsn212dcfba394f',
      'x-rapidapi-host': 'booking-com15.p.rapidapi.com'
    },
  };

  try {
    const response = await axios.request(options);
    return res.status(200).json({ data: response.data });
  } catch (error) {
    console.error(error);
    return res.status(400).json({ message: "getOneWay not found." });
  }
};

const predictFlight = async (req, res) => {
  try {
    const {
      date,
      airline,
      origin_airport,
      destination_airport,
      departure_time,
      arrival_time,
    } = req.body;

    // Basic validation before calling the model API (the model API validates again)
    if (!date || !airline || !origin_airport || !destination_airport || !departure_time || !arrival_time) {
      return res.status(400).json({ message: "Flight prediction failed.", error: "missing required fields" });
    }

    // The model API computes the distance itself from the airport coordinates (statute miles, as in training)
    const predictionData = { date, airline, origin_airport, destination_airport, departure_time, arrival_time };
    const headers = process.env.MODEL_API_KEY ? { "X-API-Key": process.env.MODEL_API_KEY } : {};

    const response = await axios.post(`${MODEL_API_URL}/predict`, predictionData, { headers, timeout: 20000 });

    return res.status(200).json({ data: response.data });
  } catch (error) {
    console.error("Prediction error:", error.message);
    return res.status(400).json({
      message: "Flight prediction failed.",
      error: error.response?.data?.error || error.message,
    });
  }
};

export { getOneWay, getOneWay7DaysMinPrice, predictFlight };
