import { Tag, Tooltip, Progress, Alert } from "antd";
import { AiOutlineClockCircle, AiOutlineRobot, AiOutlineWarning } from "react-icons/ai";
import { MdFlightTakeoff, MdFlightLand } from "react-icons/md";
import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BsStars } from "react-icons/bs";

// Class ids returned by the model API (model-api/cancel_delay_api.py) -> labels shown to the user
const DELAY_LABELS = {
  on_time_or_early: "Zamanında veya erken",
  delay_1_15: "Hafif gecikme (1-15 dk)",
  delay_16_30: "Orta gecikme (16-30 dk)",
  delay_over_30: "Ciddi gecikme (30+ dk)",
};
const DELAY_COLORS = { on_time_or_early: "green", delay_1_15: "processing", delay_16_30: "warning", delay_over_30: "error" };
const CAUSE_LABELS = { A: "Havayolu / taşıyıcı", B: "Hava durumu", C: "Ulusal hava sistemi" };

const Ticket = ({
  logo,
  airline,
  departureTime,
  duration,
  arrivalTime,
  departureCode,
  arrivalCode,
  price,
  model,
}) => {
  const [isCollapseOpen, setIsCollapseOpen] = useState(false);

  const pred = model?.data?.predictions;
  const inputs = model?.data?.inputs;
  const warnings = model?.data?.warnings || [];
  const info = model?.data?.model_info;
  const pct = (v) => Math.round((v ?? 0) * 1000) / 10; // one decimal, e.g. 3.4

  const delayClass = pred?.delay?.delay_class;
  const delayIcon = () => {
    if (delayClass === "on_time_or_early") return <MdFlightTakeoff />;
    if (delayClass === "delay_over_30") return <MdFlightLand />;
    return <AiOutlineClockCircle />;
  };

  return (
    <>
      <div className="flex max-sm:flex-col items-center gap-5 w-[70%] max-sm:w-[98%] mx-auto max-sm:mx-1 mt-6 border rounded-lg py-8 px-5">
        <div className="flex justify-between gap-10 sm:gap-14 w-full">
          <div className="flex items-center max-sm:flex-col max-sm:gap-1 sm:gap-2 md:gap-5 lg:gap-10">
            <img
              src={logo}
              alt={airline}
              className="w-12 h-12 max-sm:w-8 max-sm:h-8 md:w-16 md:h-16 object-contain"
            />
            <span className="text-base md:text-lg font-medium">{airline}</span>
          </div>
          <div className="flex flex-col w-full md:w-1/2">
            <div className="flex items-center justify-between gap-4">
              <span className="text-gray-800 font-semibold">{departureTime}</span>
              <span className="text-gray-500 text-sm hidden md:block">
                {duration}
              </span>
              <span className="text-gray-800 font-semibold">{arrivalTime}</span>
            </div>
            <div className="relative w-full mx-auto flex items-center justify-between">
              <span className="bg-white border-2 z-10 border-gray-300 rounded-full h-3 w-3"></span>
              <span className="absolute left-[10px] z-0 bg-gray-100 w-[98%] h-1 border"></span>
              <span className="bg-white border-2 z-10 border-gray-300 rounded-full h-3 w-3"></span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[16px] text-gray-600 font-semibold">
                {departureCode}
              </span>
              <span className="text-gray-500 text-sm">Direct</span>
              <span className="text-[16px] text-gray-600 font-semibold">
                {arrivalCode}
              </span>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 w-full md:w-auto items-center justify-center">
            {pred ? (
              <>
                {pred.cancel_alert ? (
                  <Tag color="error" className="flex items-center gap-2 py-0.5">
                    <AiOutlineWarning className="text-lg" />
                    İptal riski yüksek (%{pct(pred.cancelled_probability.cancelled)})
                  </Tag>
                ) : (
                  <Tag color="success" className="flex items-center gap-2 py-0.5">
                    İptal riski düşük (%{pct(pred.cancelled_probability.cancelled)})
                  </Tag>
                )}
                <Tag color={DELAY_COLORS[delayClass] || "default"} className="flex items-center gap-2 py-0.5">
                  {delayIcon()}
                  {DELAY_LABELS[delayClass]}
                </Tag>
                <Tooltip title="Genel iptal oranına kıyasla, bu uçuş için modelin hesapladığı iptal olasılığı">
                  <Tag color="blue" className="flex items-center gap-2 py-0.5">
                    <AiOutlineRobot className="text-lg" />
                    {pred.relative_risk.toFixed(1)}x göreli iptal riski
                  </Tag>
                </Tooltip>
              </>
            ) : (
              <Tag className="py-0.5">Risk tahmini alınamadı</Tag>
            )}
            {pred && (
              <Tag
                color="cyan"
                className="flex items-center gap-2 py-1 my-2 cursor-pointer hover:opacity-80"
                onClick={() => setIsCollapseOpen(!isCollapseOpen)}
              >
                <BsStars className="text-lg" />
                <span>FCDP-SFTS AI Model Tahmin Detayları</span>
              </Tag>
            )}
          </div>
        </div>
        <div className="relative flex flex-col items-center md:border-l-2 md:border-dashed md:pl-6 mt-4 md:mt-0">
          <div className="text-center">
            <h2 className="text-xl md:text-2xl font-bold flex items-baseline gap-1 justify-center">
              {price}
            </h2>
            <span className="text-gray-500 text-sm">Per person</span>
          </div>
        </div>
      </div>
      <div className="w-[70%] max-sm:w-[98%] mx-auto max-sm:mx-1">
        <AnimatePresence>
          {isCollapseOpen && pred && (
            <motion.div
              key="ai-details"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
              style={{ overflow: "hidden" }}
              className="space-y-4 p-4 bg-gray-50 border border-gray-200 mt-2 rounded-lg"
            >
              <div>
                <h2 className="text-xl text-center font-bold mb-2">
                  <span className="bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 bg-clip-text text-transparent">
                    FCDP-SFTS AI Model Tahmin Detayları
                  </span>
                </h2>
                <h3 className="font-semibold mb-2">İptal olasılığı</h3>
                <div className="flex items-center gap-4">
                  <Progress
                    type="circle"
                    percent={pct(pred.cancelled_probability.cancelled)}
                    format={(percent) => `${percent}%`}
                    strokeColor="#ff4d4f"
                  />
                  <div>
                    <p className="text-sm text-gray-600">
                      İptal olasılığı: <span className="font-bold">%{pct(pred.cancelled_probability.cancelled)}</span>
                    </p>
                    <p className="text-sm text-gray-600">
                      Tüm uçuşlardaki ortalama iptal oranı (eğitim dönemi): <span className="font-bold">%{pct(pred.base_rate)}</span>
                    </p>
                    <p className="text-sm text-gray-600">
                      Uyarı eşiği (doğrulama döneminde seçildi): <span className="font-bold">%{pct(pred.alert_threshold)}</span>
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-2">Kalkış gecikmesi (olasılık dağılımı)</h3>
                <div className="space-y-2">
                  {Object.entries(pred.delay.delay_probabilities).map(([key, value]) => (
                    <div key={key} className="flex items-center gap-2">
                      <span className="w-48 text-sm text-gray-600">{DELAY_LABELS[key]}:</span>
                      <Progress
                        percent={pct(value)}
                        size="small"
                        status={key === delayClass ? "active" : "normal"}
                        strokeColor={key === "on_time_or_early" ? "#52c41a" : "#ff4d4f"}
                      />
                    </div>
                  ))}
                  <p className="text-sm text-gray-600">
                    15 dakikadan fazla gecikme olasılığı: <span className="font-bold">%{pct(pred.delay.probability_delay_over_15_min)}</span>
                  </p>
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-2">İptal edilirse olası neden</h3>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(pred.cancellation_cause_if_cancelled).map(([key, value]) => (
                    <Tag key={key} color="blue">
                      {CAUSE_LABELS[key]}: %{pct(value)}
                    </Tag>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-2">Kullanılan veriler</h3>
                <div className="flex flex-wrap gap-2">
                  <Tag color={inputs?.weather_available ? "green" : "orange"}>
                    {inputs?.weather_available ? "Önceki günün gözlenen hava verisi kullanıldı" : "Hava verisi yok"}
                  </Tag>
                  <Tag color={inputs?.history_available ? "green" : "orange"}>
                    {inputs?.history_available ? "Son günlerin iptal/gecikme oranları kullanıldı" : "Güncel operasyon geçmişi yok (ortalama değerler)"}
                  </Tag>
                </div>
                {warnings.map((w, i) => (
                  <p key={i} className="text-xs text-gray-500 mt-1">{w}</p>
                ))}
              </div>

              <Alert
                type="info"
                showIcon
                message="Bu bir hava tahmini veya garanti değildir"
                description={
                  info
                    ? `Model, önceki günün gözlemlerine ve geçmiş oranlara dayalı bir risk göstergesidir. ${info.evaluation_period} döneminde test edilmiş iptal ROC-AUC değeri ${info.cancel_test_roc_auc.toFixed(2)}, gecikme sınıfı doğruluğu %${pct(info.delay_test_accuracy)} (her uçuşa "zamanında" demenin doğruluğu %${pct(info.delay_majority_baseline)}). Olasılıklara ek bir düzeltme uygulanmamıştır.`
                    : "Model, önceki günün gözlemlerine ve geçmiş oranlara dayalı bir risk göstergesidir."
                }
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
};

export default Ticket;
