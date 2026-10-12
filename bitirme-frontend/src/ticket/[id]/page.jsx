import { useState, useEffect } from "react";
import { DownOutlined, UserOutlined } from "@ant-design/icons";
import { Button, Dropdown, message, Space, Spin } from "antd";
import flyLogo from "../../assets/images/VF.webp";
import OtherTicketDays from "../../components/OtherTicketDays";
import Ticket from "../../components/Ticket";
import TravelEdit from "../../components/TravelEdit";
import { useLocation, useNavigate } from "react-router-dom";
import dayjs from "dayjs";
import { axiosPrivate } from "../../api/axios";

const formatDateToYYYYMMDD = (dateStr) => {
  const date = new Date(dateStr);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const extractTimeFromISO = (isoString) => {
  const date = new Date(isoString);
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit'
  });
};

const TicketDetails = () => {
  const location = useLocation();
  const { fromValue, toValue, flightInfo, flightData, flightData7DaysMinPrice } = location.state || {};
  const [isEdit, setIsEdit] = useState(false);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    const fetchCoordinatesAndPredict = async () => {
      try {
        const originAirport = flightData[0].legs[0].segments[0].origin.displayCode;
        const destAirport = flightData[0].legs[0].segments[0].destination.displayCode;

        // One prediction per listed flight. Airports are sent as IATA codes; the model API derives the distance itself.
        const predictionPromises = flightData.map(async (itinerary) => {
          const predictFlightData = {
            date: formatDateToYYYYMMDD(flightInfo.dates.departDate),
            airline: itinerary.legs[0].segments[0].marketingCarrier.alternateId,
            origin_airport: originAirport,
            destination_airport: destAirport,
            departure_time: extractTimeFromISO(itinerary.legs[0].segments[0].departure),
            arrival_time: extractTimeFromISO(itinerary.legs[0].segments[0].arrival),
          };

          try {
            const response = await axiosPrivate.post('/flight/predict', predictFlightData);
            return response.data;
          } catch (err) {
            console.error('Prediction failed for one flight:', err);
            return null; // the ticket card is shown without a risk indication
          }
        });

        const results = await Promise.all(predictionPromises);
        setPredictions(results);
      } catch (error) {
        console.error('Error fetching coordinates or making predictions:', error);
      }
    };

    if (flightData && flightData.length > 0) {
      fetchCoordinatesAndPredict();
    }
  }, [flightData, flightInfo, fromValue, toValue]);

  const items = [
    {
      label: "1st menu item",
      key: "1",
      icon: <UserOutlined />,
    },
    {
      label: "2nd menu item",
      key: "2",
      icon: <UserOutlined />,
    },
    {
      label: "3rd menu item",
      key: "3",
      icon: <UserOutlined />,
      danger: true,
    },
    {
      label: "4th menu item",
      key: "4",
      icon: <UserOutlined />,
      danger: true,
      disabled: true,
    },
  ];


  const menuProps = {
    items,
  };

  const formatDuration = (minutes) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${String(hours).padStart(2, "0")}h ${String(mins).padStart(2, "0")}m`;
  };
  const router = useNavigate();
  const handleOtherTicketSearch = async (date) => {
    try {
      setLoading(true);
      const formattedDate = dayjs(date).format("YYYY-MM-DD");
      const oneWay = await axiosPrivate.get("/flight/oneWay", {
        params: {
          from: flightInfo.fromCode,
          to: flightInfo.toCode,
          date: formattedDate,
          adults: flightInfo.passengers.adult,
          children: flightInfo.passengers.children,
          infants: flightInfo.passengers.infants,
          cabinClass: flightInfo.classType,
        },
      });
      const oneWay7DaysMinPrice = await axiosPrivate.get("/flight/oneWay7DaysMinPrice", {
        params: {
          fromId: flightInfo.fromCode.toLocaleUpperCase(),
          toId: flightInfo.toCode.toLocaleUpperCase(),
          departDate: formattedDate,
          cabinClass: flightInfo.classType.toLocaleUpperCase(),
        },
      });
      setLoading(false);
      router(`/ticket/${oneWay.data.data.data.flightsSessionId}`, {
        state: {
          fromValue,
          toValue,
          flightInfo,
          flightData: oneWay.data.data.data.itineraries,
          flightData7DaysMinPrice: oneWay7DaysMinPrice.data.data.data,
        },
      });
    } catch (error) {
      console.error("Error fetching flights:", error);
      setLoading(false);
    }
  };

  return loading ? (
    <div className="flex items-center justify-center h-screen w-full">
      <Spin />
    </div>
  )
    : (
      <div className="py-16">
        <div className="w-full py-4 px-32 max-sm:px-5 max-md:px-10 border-t-2 border-gray-100 flex justify-between items-center shadow-sm bg-white">
          {isEdit ? (
            <TravelEdit />
          ) : (
            <div className="w-full flex max-sm:flex-col gap-3 px-8 justify-between">
              <div className="flex flex-col gap-1">
                <span className="text-xl font-semibold">
                  {fromValue.split(",")[0] + "," + fromValue.split(", ")[1]} - {toValue.split(",")[0] + ", " + toValue.split(",")[1]}
                </span>
                <span className="text-gray-600">
                  {flightInfo.dates.departDate} | {flightInfo.classType.charAt(0).toUpperCase() + flightInfo.classType.slice(1)} | {flightInfo.passengers.adult} Passenger  {flightInfo.passengers.children} Child  {flightInfo.passengers.infants} Baby
                </span>
              </div>
              <button
                onClick={() => setIsEdit(true)}
                className="bg-green-500 hover:bg-green-600 transition-colors duration-300 cursor-pointer text-white font-semibold py-2 px-10 h-[45px] rounded-lg shadow"
              >
                Edit
              </button>
            </div>
          )}
        </div>
        <OtherTicketDays flightData7DaysMinPrice={flightData7DaysMinPrice} handleOtherTicketSearch={handleOtherTicketSearch} />
        <div className="w-full flex flex-col items-center gap-2 pt-6">
          <div className="grid px-3 grid-cols-5 max-sm:grid-cols-2 max-md:grid-cols-3 gap-2 justify-center">
            <Button>Önerilen</Button>
            <Button>Ücrete Göre Artan</Button>
            <Button>Uçuş Süresi</Button>
            <Dropdown menu={menuProps}>
              <Button>
                <Space>
                  Direkt/Aktarmalı
                  <DownOutlined />
                </Space>
              </Button>
            </Dropdown>
            <Dropdown menu={menuProps}>
              <Button>
                <Space>
                  Havayolu Firması
                  <DownOutlined />
                </Space>
              </Button>
            </Dropdown>
          </div>
        </div>
        {flightData.map((ticket, index) => (
          <Ticket
            key={index}
            logo={ticket.legs[0].carriers.marketing[0].logoUrl}
            airline={ticket.legs[0].carriers.marketing[0].name}
            departureTime={dayjs(ticket.legs[0].departure).format("H:mm")}
            duration={formatDuration(ticket.legs[0].durationInMinutes)}
            arrivalTime={dayjs(ticket.legs[0].arrival).format("H:mm")}
            departureCode={ticket.legs[0].origin.displayCode}
            arrivalCode={ticket.legs[0].destination.displayCode}
            price={ticket.price.formatted}
            model={predictions[index]}
          />
        ))}
      </div>
    );
};

export default TicketDetails;
