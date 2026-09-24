import { CheckCircleIcon, EyeIcon } from "@phosphor-icons/react/dist/ssr";
import { Button } from "../../ui/button";
import { Separator } from "../../ui/separator";

export type AddDatasetResponseProps = {
	datasetId: string;
	datasetVersion: string;
	detectedSample: number;
};

export function AddDatasetResponse({
	value,
	toggleModal,
	resetForm,
}: {
	value: AddDatasetResponseProps;
	toggleModal: () => void;
	resetForm: () => void;
}) {
	const responseData = [
		{ label: "Dataset ID:", value: value.datasetId, isBold: true },
		{
			label: "Target Version:",
			value: value.datasetVersion,
			color: "text-green-600 font-semibold",
		},
		{
			label: "Verified Samples:",
			value: value.detectedSample.toLocaleString(),
			isBold: true,
		},
	];

	return (
		<div className="flex flex-col items-center justify-center p-6 bg-white rounded-2xl max-w-lg mx-auto gap-6">
			{/* Success Icon */}
			<div className="p-3 bg-emerald-50 border border-emerald-200 rounded-2xl text-emerald-600 shadow-sm">
				<CheckCircleIcon className="h-8 w-8" />
			</div>

			{/* Title & Description */}
			<div className="text-center flex flex-col gap-1.5">
				<h2 className="text-xl font-bold text-gray-900 tracking-tight">
					Dataset Version Created Successfully
				</h2>
				<p className="text-xs text-muted-foreground max-w-sm">
					Manifest registered with immutable cryptographic hash and
					zero train/eval leakage. Ready for SFT training jobs.
				</p>
			</div>

			{/* Summary Box */}
			<div className="w-full flex flex-col gap-3 p-4 rounded-xl border bg-gray-50/50">
				{responseData.map((item, index) => (
					<div
						key={index}
						className="flex justify-between items-center text-sm">
						<span className="text-muted-foreground">
							{item.label}
						</span>
						<span
							className={
								item.color ||
								(item.isBold
									? "font-bold text-gray-900"
									: "text-gray-800")
							}>
							{item.value}
						</span>
					</div>
				))}

				<Separator />

				<div className="flex justify-between items-center text-sm">
					<span className="text-muted-foreground">
						Training Readiness:
					</span>
					<div className="flex items-center gap-1 bg-emerald-50 text-emerald-700 border border-emerald-200 px-2.5 py-1 rounded-full text-xs font-semibold">
						<CheckCircleIcon className="h-3.5 w-3.5" />
						Ready
					</div>
				</div>
			</div>

			{/* Action Buttons */}
			<div className="flex items-center gap-3 w-full pt-2">
				<Button
					className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white gap-2"
					onClick={() => {
						resetForm();
						toggleModal();
					}}>
					<EyeIcon className="h-4 w-4" />
					View Dataset List
				</Button>
				{/* <Button
					variant="outline"
					className="flex-1">
					Close
				</Button> */}
			</div>
		</div>
	);
}
