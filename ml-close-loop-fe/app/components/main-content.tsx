import { SidebarTrigger } from "./ui/sidebar";

export default function MainContent({children, title}: {children: React.ReactNode, title: string}) {
    return (
        <div className="w-full h-full flex flex-col gap-4 overflow-y-auto bg-stone-100 rounded-lg p-4">
            <header>
                <SidebarTrigger />
                {title}
            </header>
            <div>
                {children}
            </div>
        </div>
    )
}