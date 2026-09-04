import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from "recharts";

export default function ProfileChart({ profile }) {
  if (!profile) return null;

  const data = profile.profile.map(({ depth_m, temp_c }) => ({
    depth: depth_m,
    temp:  +temp_c.toFixed(2)
  }));

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h3 className="text-sm font-semibold text-blue-300 mb-1">
        Temperature Profile
      </h3>
      <p className="text-xs text-gray-400 mb-3">
        {profile.lat}°N {profile.lon}°E · {profile.date}
      </p>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} layout="vertical"
          margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis type="number" dataKey="temp" domain={["auto","auto"]}
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            label={{ value: "°C", position: "insideTopRight", fill: "#9ca3af", fontSize: 11 }} />
          <YAxis type="number" dataKey="depth" reversed
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            label={{ value: "Depth (m)", angle: -90, position: "insideLeft", fill: "#9ca3af", fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: "#1f2937", border: "none", borderRadius: 6 }}
            labelFormatter={(v) => `${v}m`}
            formatter={(v) => [`${v}°C`, "Temp"]}
          />
          <ReferenceLine
            y={profile.thermocline_depth}
            stroke="#f59e0b" strokeDasharray="4 2"
            label={{ value: `Thermocline ${profile.thermocline_depth}m`, fill: "#f59e0b", fontSize: 10 }}
          />
          <Line type="monotone" dataKey="temp"
            stroke="#38bdf8" strokeWidth={2} dot={{ r: 3, fill: "#38bdf8" }} />
        </LineChart>
      </ResponsiveContainer>

      <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <div className="bg-gray-700 rounded p-2">
          <p className="text-gray-400">Thermocline</p>
          <p className="text-amber-400 font-bold">{profile.thermocline_depth} m</p>
        </div>
        <div className="bg-gray-700 rounded p-2">
          <p className="text-gray-400">Thermo Grad</p>
          <p className="text-amber-400 font-bold">{profile.thermocline_grad} °C/m</p>
        </div>
      </div>
    </div>
  );
}
