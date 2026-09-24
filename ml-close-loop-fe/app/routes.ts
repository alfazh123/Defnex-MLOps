import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
	index("routes/home.tsx"),
	route("chat", "routes/chat.tsx"),
	route("dataset", "routes/dataset/dataset.tsx"),
	route("dataset/:id", "routes/dataset/dataset-detail.tsx"),
	route(
		"dataset/:id/validation-history",
		"routes/dataset/validation-history-dataset.tsx",
	),
	route("train-run", "routes/train-run.tsx"),
	route("model-registry", "routes/model-registry.tsx"),

	route("auth/login", "routes/auth/login.tsx"),
	route("auth/signin", "routes/auth/signin.tsx"),
	// route("settings", "routes/settings.tsx"),
] satisfies RouteConfig;
