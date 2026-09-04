import { useState } from "react";
import MapView from "./components/Map/MapView";
import ProfileChart from "./components/Charts/ProfileChart";
import ControlPanel from "./components/Panels/ControlPanel";

export default function App() {
  const [selectedPoint, setSelectedPoint] = useState({ lat: 15.0, lon: 80.0 });
  const [date, setDate] = useState("2022-06-01");
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      // Generate synthetic 64x64 patch (demo — replace with real satellite data)
      const patch = Array.from({ length: 64 }, () =>
        Array.from({ length: 64 }, () => 28 + Math.random() * 4 - 2)
      );
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sst: patch, sss: patch, ssh: patch, u: patch, v: patch,
          lat: selectedPoint.lat,
          lon: selectedPoint.lon,
          date
        })
      });
      const data = await res.json();
      setProfile(data);
    } catch (e) {
      setError("API error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 border-b border-blue-900 px-6 py-3 flex items-center gap-3">
        <div className="w-3 h-3 rounded-full bg-blue-400 animate-pulse" />
        <h1 className="text-lg font-bold tracking-wide text-blue-300">
          OceanEmbed
        </h1>
        <span className="text-xs text-gray-500 ml-2">
          Subsurface Temperature Reconstruction · NIO
        </span>
      </header>

      {/* Main */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Map */}
        <div className="flex-1 relative">
          <MapView
            selectedPoint={selectedPoint}
            onPointSelect={setSelectedPoint}
          />
        </div>

        {/* Right: Panel */}
        <div className="w-80 bg-gray-900 border-l border-gray-800 flex flex-col gap-4 p-4 overflow-y-auto">
          <ControlPanel
            lat={selectedPoint.lat}
            lon={selectedPoint.lon}
            date={date}
            onDateChange={setDate}
            onPredict={handlePredict}
            loading={loading}
          />
          {error && (
            <div className="text-red-400 text-sm bg-red-900/20 rounded p-2">
              {error}
            </div>
          )}
          {profile && <ProfileChart profile={profile} />}
        </div>
      </div>
    </div>
  );
}
