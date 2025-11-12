import { useEffect, useState } from "react";
import { ClipLoader } from "react-spinners";
import "./App.css";
import { MapContainer, TileLayer, Polygon, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import Papa from "papaparse";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const DEFAULT_BOUNDARY_URL = "/Boundaries/custom_boundary.csv"; // adjust to your static file

function App() {
	const [file, setFile] = useState(null);
	const [error, setError] = useState(null);
	const [holdmode, setHoldmode] = useState("BADA_Cruise");
	const [saveopt2, setSaveopt2] = useState("All");
	const [changeSpeed, setChangeSpeed] = useState(false);

	// boundary state
	const [boundaryFile, setBoundaryFile] = useState(null);
	const [boundaryData, setBoundaryData] = useState(null);

	// tma boundary inputs
	const [inputTmaDistance, setInputTmaDistance] = useState(40);
	const [inputNPoints, setInputNPoints] = useState(36);
	const [inputCenterLat, setInputCenterLat] = useState(1.35019);
	const [inputCenterLon, setInputCenterLon] = useState(103.994003);

	// tma boundary states
	const [tmaDistance, setTmaDistance] = useState(40); // default 40 NM
	const [nPoints, setNPoints] = useState(36); // default 36 points
	const [centerLat, setCenterLat] = useState(1.35019);
	const [centerLon, setCenterLon] = useState(103.994003);

	// loading state
	const [loading, setLoading] = useState(false);

	useEffect(() => {
		// clear boundary when saveopt2 changes
		setBoundaryFile(null);
		setBoundaryData(null);

		if (saveopt2 === "Custom") {
			// Load default boundary CSV on mount of Custom saveopt2
			if (!boundaryFile || boundaryFile.name !== "custom_boundary.csv") {
				fetch(DEFAULT_BOUNDARY_URL)
					.then((response) => response.blob())
					.then((blob) => {
						console.log("Fetched default boundary CSV blob:", blob);
						const defaultFile = new File([blob], "custom_boundary.csv", {
							type: "text/csv",
						});
						parseBoundaryCSV(defaultFile);
					});
			}
		} else if (saveopt2 === "TMA") {
			if (!boundaryFile || boundaryFile.name !== "tmaboundaries.csv") {
				// Generate TMA boundary when TMA option is selected
				handleDrawTMA();
			}
		}
	}, [saveopt2]);

	// Debugging: log boundaryData whenever it changes
	// useEffect(() => {
	// 	console.log("Boundary data updated:", boundaryData);
	// }, [boundaryData]);

	const parseBoundaryCSV = (file) => {
		Papa.parse(file, {
			header: true,
			dynamicTyping: true,
			complete: (results) => {
				// remove rows with null latitude or longitude
				const data = results.data.filter(
					(row) => row.latitude != null && row.longitude != null
				);
				// Check that longitude and latitude columns exist
				if (!("longitude" in data[0]) || !("latitude" in data[0])) {
					setError(
						"Boundary CSV must contain 'longitude' and 'latitude' columns."
					);
					return;
				}
				console.log("Parsed boundary data:", data);
				// Check that all values are numeric
				const allValid = data.every(
					(row) =>
						typeof row.latitude === "number" &&
						!isNaN(row.latitude) &&
						typeof row.longitude === "number" &&
						!isNaN(row.longitude)
				);
				if (!allValid) {
					setError("All longitude and latitude values must be numeric.");
					return;
				}

				const parsedBoundaryData = data.map((row) => ({
					latitude: row.latitude,
					longitude: row.longitude,
				}));

				setBoundaryFile(file);
				setBoundaryData(parsedBoundaryData);
				setError(null);
			},
			error: (err) => {
				console.error("Error parsing boundary CSV:", err);
			},
		});
	};

	const downloadCurrentBoundary = () => {
		if (!boundaryFile) {
			setError("No boundary file to download.");
			return;
		}
		const url = window.URL.createObjectURL(boundaryFile);
		const a = document.createElement("a");
		a.href = url;
		a.download = boundaryFile.name;
		a.click();
		window.URL.revokeObjectURL(url);
	};

	// set ChangeSpeed variable
	const handleToggle = () => setChangeSpeed((prev) => !prev);

	const resetTMA = () => {
		setInputTmaDistance(40);
		setInputNPoints(36);
		setInputCenterLat(1.35019);
		setInputCenterLon(103.994003);
	};

	const generateTMA = (distance, nPoints, centerLat, centerLon) => {
		// generate TMA boundary points
		const points = [];
		const radiusDeg = distance / 60;
		for (let i = 0; i < nPoints; i++) {
			const angle = (i / nPoints) * 2 * Math.PI;
			const lat = centerLat + radiusDeg * Math.sin(angle);
			const lon =
				centerLon +
				(radiusDeg * Math.cos(angle)) / Math.cos((centerLat * Math.PI) / 180);
			points.push({ latitude: lat, longitude: lon });
		}

		const csvContent =
			"latitude,longitude\n" +
			points.map((p) => `${p.latitude},${p.longitude}`).join("\n");
		const csvFile = new File([csvContent], "tmaboundaries.csv", {
			type: "text/csv",
		});

		setBoundaryFile(csvFile);
		setBoundaryData(points);
	};

	// handle TMA boundary creation here
	const handleDrawTMA = () => {
		// validate inputs before drawing
		let validDistance = parseFloat(inputTmaDistance);
		let validNPoints = parseInt(inputNPoints);
		let validCenterLat = parseFloat(inputCenterLat);
		let validCenterLon = parseFloat(inputCenterLon);

		if (isNaN(validDistance) || validDistance < 2) {
			setError("TMA Distance must be more than 1 NM.");
			setInputTmaDistance(tmaDistance);
			return;
		}
		if (isNaN(validNPoints) || validNPoints < 3) {
			setError("Number of Points must be an integer >= 3.");
			setInputNPoints(nPoints);
			return;
		}
		if (isNaN(validCenterLat) || validCenterLat < -90 || validCenterLat > 90) {
			setError("Center Latitude must be between -90 and 90.");
			setInputCenterLat(centerLat);
			return;
		}
		if (
			isNaN(validCenterLon) ||
			validCenterLon < -180 ||
			validCenterLon > 180
		) {
			setError("Center Longitude must be between -180 and 180.");
			setInputCenterLon(centerLon);
			return;
		}

		// update states
		setTmaDistance(validDistance);
		setNPoints(validNPoints);
		setCenterLat(validCenterLat);
		setCenterLon(validCenterLon);
		setError(null);

		console.log("Drawing TMA with:", {
			validDistance,
			validNPoints,
			validCenterLat,
			validCenterLon,
		});

		generateTMA(validDistance, validNPoints, validCenterLat, validCenterLon);
	};

	const handleUpload = async () => {
		setError(null);
		if (!file) return;

		const fileInput = document.getElementById("main-file-input");
		const freshFile = fileInput?.files[0];

		setLoading(true); // start loading indicator
		const formData = new FormData();
		formData.append("file", freshFile);
		formData.append("holdmode", holdmode);
		formData.append("saveopt2", saveopt2);
		formData.append("change_speed", changeSpeed);

		// add if custom boundary is used
		if ((saveopt2 === "Custom") | "TMA" && boundaryFile) {
			formData.append("boundary_file", boundaryFile);
		}

		try {
			const response = await fetch(`${API_URL}/process-csv`, {
				method: "POST",
				body: formData,
			});

			if (!response.ok) {
				const errorData = await response.json();
				setError(
					`Error ${response.status}: ${errorData.detail || "Unknown error"}`
				);
				console.error("Error response:", errorData);
				return;
			}
			const blob = await response.blob();
			console.log("Received blob:", blob);
			const url = window.URL.createObjectURL(blob);

			const a = document.createElement("a");
			a.href = url;
			a.download = "output.csv";
			a.click();
			window.URL.revokeObjectURL(url);
		} catch (error) {
			console.error("Error:", error);
			setError(error.message || "Something went wrong");
		} finally {
			setLoading(false); // stop loading indicator
		}
	};

	return (
		<div style={{ padding: "2rem" }}>
			<h1>Fuel Calculator App</h1>
			<input
				id="main-file-input"
				type="file"
				accept=".csv"
				onChange={(e) => setFile(e.target.files[0])}
				style={{ marginBottom: "1rem" }}
			/>
			<br />
			{/* Holdmode Dropdown */}
			<label>
				Hold Mode:
				<select value={holdmode} onChange={(e) => setHoldmode(e.target.value)}>
					<option value="BADA_Cruise">BADA_Cruise</option>
					<option value="BADA">BADA</option>
					<option value="NATS">NATS</option>
				</select>
			</label>
			<br />
			<br />
			{/* SaveOpt2 Dropdown */}
			<label>
				Get Boundary in:
				<select value={saveopt2} onChange={(e) => setSaveopt2(e.target.value)}>
					<option value="All">All</option>
					<option value="TMA">TMA</option>
					<option value="Custom">Custom</option>
				</select>
			</label>
			{/* TMA Boundary Upload */}
			{saveopt2 === "TMA" && (
				<div className="vertical-stack">
					<p>TMA Boundary:</p>
					<label>
						Distance (NM)
						<input
							type="number"
							value={inputTmaDistance}
							onChange={(e) => setInputTmaDistance(e.target.value)}
						/>
					</label>
					<label>
						Number of Points
						<input
							type="number"
							value={inputNPoints}
							onChange={(e) => setInputNPoints(e.target.value) || 3}
							min={3}
						/>
					</label>
					<label>
						Center Latitude
						<input
							type="number"
							value={inputCenterLat}
							onChange={(e) => setInputCenterLat(e.target.value)}
						/>
					</label>
					<label>
						Center Longitude
						<input
							type="number"
							value={inputCenterLon}
							onChange={(e) => setInputCenterLon(e.target.value)}
						/>
					</label>
					<button onClick={() => handleDrawTMA()}> Draw & Set Boundary </button>
					<button onClick={() => resetTMA()}> Reset Values </button>
					<button onClick={downloadCurrentBoundary} disabled={!boundaryFile}>
						Download TMA Boundary
					</button>
				</div>
			)}
			<br />

			{/* Custom Boundary Upload */}
			{saveopt2 === "Custom" && (
				<div className="vertical-stack">
					<p>Custom Boundary (default loaded, can upload new):</p>
					<input
						type="file"
						accept=".csv"
						onChange={(e) => parseBoundaryCSV(e.target.files[0])}
					/>
					<br />
					<button onClick={downloadCurrentBoundary}>
						Download Current Boundary
					</button>
					<p>{boundaryFile && `Current boundary: ${boundaryFile.name}`}</p>
					<p>Boundary File just requires "longitude" and "latitude" columns</p>
				</div>
			)}
			<br />
			<br />
			<label className="toggle-switch">
				<input type="checkbox" checked={changeSpeed} onChange={handleToggle} />
				<span className="slider"></span>
				<span className="label-text">
					{changeSpeed ? "CHANGE SPEED ON" : "CHANGE SPEED OFF"}
				</span>
			</label>
			<br />
			<button onClick={handleUpload}>Upload & Process CSV</button>
			{loading && <ClipLoader size={20} color="#123abc" />}
			<p style={{ color: "red", minHeight: "1.5em" }}>{error || ""}</p>

			{/* Map is always visible */}
			<div style={{ marginTop: "1rem", marginBottom: "1rem" }}>
				<h3>Boundary Preview:</h3>
				<MapContainer
					center={[centerLat, centerLon]}
					zoom={8}
					style={{ height: "400px", width: "100%" }}
				>
					<TileLayer
						attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
						url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
					/>

					{/* Only draw polygon when boundaryData exists */}
					{boundaryData && boundaryData.length > 0 && (
						<Polygon
							positions={boundaryData.map((point) => [
								point.latitude,
								point.longitude,
							])}
							pathOptions={{
								color: "blue",
								fillColor: "lightblue",
								fillOpacity: 0.3,
							}}
						/>
					)}

					{/* Auto-fit bounds when boundary changes */}
					<BoundaryFitter boundaryData={boundaryData} />
				</MapContainer>
			</div>
		</div>
	);
}

// Component to auto-fit bounds when boundary changes
function BoundaryFitter({ boundaryData }) {
	const map = useMap();

	useEffect(() => {
		if (boundaryData && boundaryData.length > 0) {
			// Convert boundary data to [lat, lon] pairs
			const bounds = boundaryData.map((point) => [
				point.latitude,
				point.longitude,
			]);
			// Tell map to zoom/pan to show all these points
			map.fitBounds(bounds, { padding: [50, 50] });
		}
	}, [boundaryData, map]);

	return null;
}

export default App;
