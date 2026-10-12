import axios from "axios";

const MODEL_API_URL = "http://localhost:5050";

const getCountries = async (req, res) => {
  const options = {
    method: "GET",
    url: "https://country-state-city-search-rest-api.p.rapidapi.com/allcountries",
    headers: {
      "x-rapidapi-key": process.env.RAPIDAPI_KEY,
      "x-rapidapi-host": "country-state-city-search-rest-api.p.rapidapi.com",
    },
  };

  try {
    const response = await axios.request(options);
    return res.status(200).json({ data: response.data });
  } catch (error) {
    console.error(error);
    return res.status(400).json({ message: "allcountries not found." });
  }
};

const getCountryUs = async (req, res) => {
  try {
    return res.status(200).json({
      data: [{
        name: "United States",
        isoCode: "US"
      }]
    });
  } catch (error) {
    console.error(error);
    return res.status(400).json({ message: "getCountryUs not found." });
  }
};

const getCities = async (req, res) => {
  try {
    const response = await axios.get(`${MODEL_API_URL}/cities`);
    return res.status(200).json({ data: response.data.cities });
  } catch (error) {
    console.error(error);
    return res.status(400).json({ message: "getCities not found." });
  }
};

const getCitiesJson = async (req, res) => {
  try {
    const response = await axios.get(`${MODEL_API_URL}/cities`);
    const cities = response.data.cities.map(city => ({ name: city }));
    return res.status(200).json({ data: cities });
  } catch (error) {
    console.error(error);
    return res.status(400).json({ message: "getCities not found." });
  }
};

const getAirports = async (req, res) => {
  const { city } = req.query;

  const options = {
    method: "GET",
    url: "https://flights-sky.p.rapidapi.com/flights/auto-complete",
    params: {
      query: city,
      placeTypes: "AIRPORT",
    },
    headers: {
      "x-rapidapi-key": process.env.RAPIDAPI_KEY,
      "x-rapidapi-host": "flights-sky.p.rapidapi.com",
    },
  };

  try {
    const response = await axios.request(options);
    return res.status(200).json({ data: response.data });
  } catch (error) {
    console.error(error);
    return res.status(400).json({ message: "getCities not found." });
  }
};

const getCityPhotos = async (req, res) => {
  const { cityName } = req.query;
  const unsplashApi = axios.create({
    baseURL: "https://api.unsplash.com",
    headers: {
      Authorization: `Client-ID ${process.env.UNSPLASH_ACCESS_KEY}`,
    },
  });

  try {
    const response = await unsplashApi.get("/search/photos", {
      params: {
        query: cityName,
        per_page: 5,
      },
    });
    return res.status(200).json({ data: response.data.results });
  } catch (error) {
    console.error(
      `${cityName} - error while fetching photos:`,
      error
    );
    return res.status(400).json({ message: "getCityPhotos not found." });
  }
};

const getAirportByCode = async (req, res) => {
  const { code } = req.query;
  const options = {
    method: 'GET',
    url: `https://iata-airports.p.rapidapi.com/airports/${code}/`,
    headers: {
      'x-rapidapi-key': process.env.RAPIDAPI_KEY,
      'x-rapidapi-host': 'iata-airports.p.rapidapi.com'
    }
  };

  try {
    const response = await axios.request(options);
    return res.status(200).json({ data: response.data });
  } catch (error) {
    console.error(error);
    return res.status(400).json({ message: 'getAirportByCode not found.' });
  }
};

export { getCountries, getCountryUs, getCities, getCitiesJson, getAirports, getCityPhotos, getAirportByCode };
