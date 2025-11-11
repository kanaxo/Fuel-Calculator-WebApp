import { useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
	const [file, setFile] = useState(null);

	const handleUpload = async () => {
		if (!file) return;

		const formData = new FormData();
		formData.append("file", file);

		const response = await fetch(`${API_URL}/process-csv`, {
			method: "POST",
			body: formData,
		});

		const blob = await response.blob();
		const url = window.URL.createObjectURL(blob);

		const a = document.createElement("a");
		a.href = url;
		a.download = "output.csv";
		a.click();
		window.URL.revokeObjectURL(url);
	};

	return (
		<div style={{ padding: "2rem" }}>
			<h1>Fuel Calculator App</h1>
			<input
				type="file"
				accept=".csv"
				onChange={(e) => setFile(e.target.files[0])}
				style={{ marginBottom: "1rem" }}
			/>
			<br />
			<button onClick={handleUpload}>Upload & Process CSV</button>
		</div>
	);
}

export default App;
