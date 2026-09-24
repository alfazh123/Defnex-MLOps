import { ArrowsClockwiseIcon } from "@phosphor-icons/react/dist/ssr";
import { Button } from "./button";
import { Dialog, DialogTrigger } from "./dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";

export default function Header({
	title,
	description,
	newKnowledge,
	modalAddKnowledge,
	sft,
	modalSft,
}: {
	title: string;
	description?: string;
	newKnowledge?: boolean;
	modalAddKnowledge?: () => void;
	sft?: boolean;
	modalSft?: () => void;
}) {
	return (
		<div className="sticky top-0 flex sm:flex-nowrap flex-wrap w-full border-b border-stone-300 justify-between items-center p-2 bg-sidebar mb-4 z-10 rounded-t-lg">
			<div className="flex items-center p-2">
				{/* <SidebarTrigger />
				<hr className="rotate-90 bg-slate-500 h-0.5 w-5" /> */}
				<div>
					<h2 className="text-lg font-semibold">{title}</h2>
					{description && (
						<p className="text-sm text-accent-foreground">
							{description}
						</p>
					)}
				</div>
			</div>

			{newKnowledge && (
				<Dialog>
					<form>
						<DialogTrigger
							onClick={modalAddKnowledge}
							render={
								<Button variant="default">Add Dataset</Button>
							}
						/>
					</form>
				</Dialog>
			)}

			{sft && (
				<div className="flex gap-2 items-center">
					<Tooltip>
						<TooltipTrigger
							render={
								<Button variant="outline">
									<ArrowsClockwiseIcon
										className="h-4 w-4"
										weight="fill"
									/>
								</Button>
							}
						/>
						<TooltipContent>
							<p>Refresh worker</p>
						</TooltipContent>
					</Tooltip>
					<Dialog>
						<form>
							<DialogTrigger
								onClick={modalSft}
								render={
									<Button variant="default">
										Launch SFT Run
									</Button>
								}
							/>
						</form>
					</Dialog>
				</div>
			)}
		</div>
	);
}
