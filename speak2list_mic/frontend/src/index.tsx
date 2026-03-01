import React from "react";
import ReactDOM from "react-dom/client";
import { withStreamlitConnection } from "streamlit-component-lib";
import MicButton from "./MicButton";

const Connected = withStreamlitConnection(MicButton);
ReactDOM.createRoot(document.getElementById("root")!).render(<Connected />);