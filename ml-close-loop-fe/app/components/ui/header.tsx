import { Button } from "./button";
import { Dialog, DialogContent, DialogTrigger } from "./dialog";
import { SidebarTrigger } from "./sidebar";

export default function Header({title, newKnowledge, modalAddKnowledge, sft, modalSft}: {title: string, newKnowledge?: boolean, modalAddKnowledge?: () => void, sft?: boolean, modalSft?: () => void}) {
    return (
        <div className="sticky top-0 flex w-full border-b border-stone-300 justify-between p-2 bg-sidebar mb-4">
            <div className="flex items-center p-2">
                <SidebarTrigger />
                <hr className="rotate-90 bg-slate-500 h-0.5 w-5" />
                <h2 className="text-lg font-semibold">{title}</h2>
            </div>

            {newKnowledge && (
                <Dialog>
                    <form>
                        <DialogTrigger onClick={modalAddKnowledge} render={<Button variant="default">Add Dataset</Button>} />
                    </form>
                </Dialog>
            )}

            {sft && (
                <Dialog>
                    <form>
                        <DialogTrigger onClick={modalSft} render={<Button variant="default">Launch SFT Run</Button>} />
                    </form>
                </Dialog>
            )}
        </div>
    )
}