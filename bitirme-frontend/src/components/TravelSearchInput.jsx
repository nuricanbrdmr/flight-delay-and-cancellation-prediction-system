import { AutoComplete, Input } from "antd";
import { useState, useEffect } from "react";
import { axiosPrivate } from "../api/axios";

const renderItem = (title, flagCode) => (
  <div
    style={{
      display: "flex",
      padding: "0 10px",
      justifyContent: "space-between",
    }}
  >
    {title}
    <span>
      {flagCode && (
        <img
          src={`https://flagsapi.com/${flagCode}/flat/24.png`}
          width={24}
          height={28}
          alt="flag"
          onError={(e) =>
          (e.target.src =
            "https://upload.wikimedia.org/wikipedia/commons/0/03/Flag_white.svg")
          }
        />
      )}
    </span>
  </div>
);

const TravelSearchInput = ({
  title,
  value,
  setValue,
  setCode,
  leftMargin,
  countries,
}) => {
  const [options, setOptions] = useState([]);
  const [initialOptions, setInitialOptions] = useState({
    ulkeler: [],
    sehirler: [],
    havalimanlari: [],
  });
  const [loading, setLoading] = useState(false);
  const [selectedCountry, setSelectedCountry] = useState(null);
  const [selectedCity, setSelectedCity] = useState(null);
  const [open, setOpen] = useState(false);
  const [selectionStep, setSelectionStep] = useState("country");

  useEffect(() => {
    if (countries && countries.length > 0) {
      const countryOptions = countries.map((country) => ({
        label: renderItem(country.name, country.isoCode),
        value: country.name,
        isoCode: country.isoCode,
      }));
      setOptions(countryOptions);
      setInitialOptions((prev) => ({
        ...prev,
        ulkeler: countryOptions,
      }));
    }
  }, [countries]);

  const fetchCities = async (title) => {
    try {
      setLoading(true);
      const { data } = await axiosPrivate.get(`/location/cities`, {
        params: { flightDirection: title },
      });
      const cityOptions = data.data.map((city) => ({
        label: renderItem(city.name),
        value: city.name,
      }));
      setOptions(cityOptions);
      setInitialOptions((prev) => ({
        ...prev,
        sehirler: cityOptions,
      }));
      setLoading(false);
      setOpen(true);
      setSelectionStep("city");
    } catch (error) {
      console.error("Error fetching cities:", error);
      setLoading(false);
    }
  };

  const fetchAirports = async (city) => {
    try {
      setLoading(true);
      const { data } = await axiosPrivate.get(`/location/airports`, {
        params: { city: city },
      });
      const airportOptions = data.data.data.map((airport) => ({
        label: renderItem(airport.presentation.suggestionTitle + " Airport"),
        value: airport.presentation.suggestionTitle + " Airport",
        code: airport.presentation.skyId,
      }));
      setOptions(airportOptions);
      setInitialOptions((prev) => ({
        ...prev,
        havalimanlari: airportOptions,
      }));
      setLoading(false);
      setOpen(true);
      setSelectionStep("airport");
    } catch (error) {
      console.error("Error fetching airports:", error);
      setLoading(false);
    }
  };

  const onSearch = (searchText) => {
    setOpen(true);
    const searchTerms = searchText.split(",");
    const searchTerm = searchTerms[searchTerms.length - 1].trim();

    switch (selectionStep) {
      case "country":
        const filteredCountries = initialOptions.ulkeler.filter((option) =>
          option.value.toLowerCase().includes(searchTerm.toLowerCase())
        );
        setOptions(filteredCountries);
        break;
      case "city":
        const filteredCities = initialOptions.sehirler.filter((option) =>
          option.value.toLowerCase().includes(searchTerm.toLowerCase())
        );
        setOptions(filteredCities);
        break;
      case "airport":
        const filteredAirports = initialOptions.havalimanlari.filter((option) =>
          option.value.toLowerCase().includes(searchTerm.toLowerCase())
        );
        setOptions(filteredAirports);
        break;
    }
  };

  const onSelect = (data, option) => {
    switch (selectionStep) {
      case "country":
        setSelectedCountry(option.isoCode);
        fetchCities(title);
        setValue(data + ",");
        break;
      case "city":
        setSelectedCity(data);
        fetchAirports(data);
        setValue(value.split(",")[0] + "," + data + ",");
        break;
      case "airport":
        const fullSelection = `${data},${value.split(",")[1]},${value.split(",")[0]
          }`;
        setValue(fullSelection);
        setCode(option.code); // Havalimanı kodu gönderiliyor
        setOpen(false);
        break;
    }
  };

  const resetSelection = () => {
    setSelectedCountry(null);
    setSelectedCity(null);
    setOptions(initialOptions.ulkeler);
    setValue("");
    setCode(""); // Kodu sıfırla
    setOpen(true);
    setSelectionStep("country");
  };

  const getPlaceholder = () => {
    switch (selectionStep) {
      case "country":
        return "Select Country";
      case "city":
        return "Select City";
      case "airport":
        return "Select Airport";
      default:
        return "Select Country";
    }
  };

  return (
    <AutoComplete
      popupClassName="certain-category-search-dropdown"
      popupMatchSelectWidth={350}
      value={value}
      options={options}
      style={{ width: "100%", height: "70px" }}
      onSearch={onSearch}
      onSelect={onSelect}
      onChange={setValue}
      loading={loading}
      size="large"
      open={open}
      onBlur={() => setOpen(false)}
    >
      <div style={{ position: "relative", width: "100%" }}>
        <div
          style={{
            position: "absolute",
            zIndex: "1",
            top: "4px",
            left: `${leftMargin}px`,
            fontSize: "12px",
            color: "#999",
          }}
        >
          {title}
        </div>
        <Input
          value={value}
          placeholder={getPlaceholder()}
          onChange={(e) => setValue(e.target.value)}
          onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)}
          style={{
            paddingTop: "20px",
            paddingLeft: `${leftMargin}px`,
            paddingBottom: "10px",
            fontSize: "20px",
          }}
        />
        {(selectedCountry || selectedCity) && (
          <button
            onClick={resetSelection}
            style={{
              position: "absolute",
              right: "15px",
              top: "20%",
              transform: "translateY(-50%)",
              fontSize: "12px",
              color: "#007bff",
              background: "none",
              border: "none",
              cursor: "pointer",
            }}
          >
            Geri
          </button>
        )}
      </div>
    </AutoComplete>
  );
};

export default TravelSearchInput;
