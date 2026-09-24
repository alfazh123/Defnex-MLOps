import Header from "~/components/ui/header";
import type { Route } from "./+types/home";

export function meta({}: Route.MetaArgs) {
	return [
		{ title: "New React Router App" },
		{ name: "description", content: "Welcome to React Router!" },
	];
}

export default function Home() {
	return (
		<>
			{/* <div className="w-full h-full flex flex-col overflow-y-auto bg-sidebar rounded-lg"> */}
			{/* <header>
              <SidebarTrigger />
              {title}
            </header> */}
			<Header title="Dashboard" />
			<div className="pl-4 pr-4 pb-4"></div>
			{/* </div> */}
		</>
	);
}
