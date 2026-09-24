import {
    FieldGroup,
} from "../../ui/field";
import { Button } from "../../ui/button";
import { ScrollArea } from "~/components/ui/scroll-area";
import {
	CircleWavyCheckIcon,
	ClockIcon,
	EyeIcon,
} from "@phosphor-icons/react/dist/ssr";

export type StepResponseValueProps = {
    runSftId: string;
}

export function SftStepResponse({value, toggleModal, resetForm}: {value: StepResponseValueProps, toggleModal: () => void, resetForm: () => void}) {
    return (
		<>
			<ScrollArea className="max-h-125">
				<FieldGroup>
					<div className="flex flex-col items-center justify-center text-center py-6 gap-6">
						{/* Ikon Sukses Tengah */}
						<div className="flex items-center justify-center h-16 w-16 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-600 shadow-sm">
							<CircleWavyCheckIcon className="h-8 w-8" />
						</div>

						{/* Teks Judul & Deskripsi */}
						<div className="flex flex-col gap-2 max-w-md">
							<h2 className="text-xl font-bold text-black">
								Training Run Queued
							</h2>
							<p className="text-xs text-muted-foreground leading-relaxed">
								An immutable training configuration and
								execution record has been created. The run will
								progress through QUEUED → RUNNING → EVALUATING
								automatically.
							</p>
						</div>

						{/* Kotak Detail Run ID & Status */}
						<div className="w-full flex flex-col gap-3 p-4 rounded-xl border bg-white text-left shadow-sm">
							<div className="flex justify-between items-center text-xs">
								<span className="text-muted-foreground font-medium">
									Run ID:
								</span>
								<span className="font-mono font-semibold text-blue-600">
									{value.runSftId}
								</span>
							</div>
							<div className="flex justify-between items-center text-xs pt-3 border-t">
								<span className="text-muted-foreground font-medium">
									Status:
								</span>
								<div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-amber-200 bg-amber-50 text-amber-700 font-medium">
									<ClockIcon className="h-3.5 w-3.5" />
									Queued
								</div>
							</div>
						</div>
					</div>
				</FieldGroup>
			</ScrollArea>

			<div className="flex justify-between items-center gap-2 pt-4 border-t mt-4">
				<Button
					className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white gap-2"
					onClick={() => {
						resetForm();
						toggleModal();
					}}>
					<EyeIcon className="h-4 w-4" />
					View Dataset List
				</Button>
				<Button
					variant="outline"
					onClick={() => {
						resetForm();
						toggleModal();
					}}
					className="flex-1">
					Close
				</Button>
			</div>
		</>
	);
}