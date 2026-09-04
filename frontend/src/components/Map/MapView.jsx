import { MapContainer, TileLayer, Marker, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Fix default marker icon
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl:       "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl:     "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

function ClickHandler({ onPointSelect }) {
  useMapEvents({
    click(e) {
      onPointSelect({ lat: +e.latlng.lat.toFixed(2), lon: +e.latlng.lng.toFixed(2) });
    }
  });
  return null;
}

export default function MapView({ selectedPoint, onPointSelect }) {
  // NIO bounds: 5N-30N, 45E-105E
  const center = [15, 75];

  return (
    <MapContainer
      center={center}
      zoom={4}
      style={{ height: "100%", width: "100%", background: "#0a0f1e" }}
      maxBounds={[[0, 40], [35, 110]]}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="&copy; OpenStreetMap contributors"
      />
      <ClickHandler onPointSelect={onPointSelect} />
      {selectedPoint && (
        <Marker position={[selectedPoint.lat, selectedPoint.lon]} />
      )}
    </MapContainer>
  );
}
