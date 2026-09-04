export default function ControlPanel({ lat, lon, date, onDateChange, onPredict, loading }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 flex flex-col gap-4">
      <h2 className="text-sm font-semibold text-blue-300 uppercase tracking-wider">
        Query Point
      </h2>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-gray-400">Latitude</label>
          <div className="mt-1 bg-gray-700 rounded px-3 py-2 text-sm font-mono text-white">
            {lat}°N
          </div>
        </div>
        <div>
          <label className="text-xs text-gray-400">Longitude</label>
          <div className="mt-1 bg-gray-700 rounded px-3 py-2 text-sm font-mono text-white">
            {lon}°E
          </div>
        </div>
      </div>

      <div>
        <label className="text-xs text-gray-400">Date</label>
        <input
          type="date"
          value={date}
          onChange={e => onDateChange(e.target.value)}
          min="2022-01-01" max="2022-12-31"
          className="mt-1 w-full bg-gray-700 rounded px-3 py-2 text-sm text-white
                     border border-gray-600 focus:border-blue-500 focus:outline-none"
        />
      </div>

      <p className="text-xs text-gray-500">
        Click on map to select a point in the North Indian Ocean
      </p>

      <button
        onClick={onPredict}
        disabled={loading}
        className="w-full py-2.5 rounded-lg font-semibold text-sm
                   bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600
                   disabled:cursor-not-allowed transition-colors"
      >
        {loading ? "Predicting..." : "Predict Profile"}
      </button>
    </div>
  );
}
