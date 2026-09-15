import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
	index("routes/home.tsx"),
	route("chat", "routes/chat.tsx"),
	route("new-knowledge", "routes/new-knowledge.tsx"),
	route("train-run", "routes/train-run.tsx"),
	route("model-registry", "routes/model-registry.tsx"),

	route("auth/login", "routes/login.tsx"),
	// route("settings", "routes/settings.tsx"),
] satisfies RouteConfig;
