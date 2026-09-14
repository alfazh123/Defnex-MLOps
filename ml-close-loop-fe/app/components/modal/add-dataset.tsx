import { Check, XIcon } from "lucide-react";
import { Button } from "../ui/button";
import {
	Dialog,
	DialogClose,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "../ui/dialog";

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { useState } from "react";
import clsx from "clsx";
import { DatasetStepTwo } from "./dataset-steps/dataset-step-two";
import {
	DatasetStepOne,
	type StepOneValue,
} from "./dataset-steps/dataset-step-one";
import { ScrollArea } from "../ui/scroll-area";
import { DatasetStepThree } from "./dataset-steps/dataset-step-three";
import { DatasetStepFour } from "./dataset-steps/dataset-step-four";
import { DatasetStepFive } from "./dataset-steps/dataset-step-five";
import { AddDatasetResponse } from "./dataset-steps/add-dataset-response";

const steps = [
	{ id: 1, name: "Source", description: "File / HF" },
	{ id: 2, name: "Dataset Info", description: "Target & Server" },
	{ id: 3, name: "Schema", description: "Format & Fields" },
	{ id: 4, name: "Validation", description: "Gate & Privew" },
	{ id: 5, name: "Review", description: "Immutable Lock" },
];

export default function AddDatasetModal({
	isOpen,
	toggleModal,
	datasetFormats,
}: {
	isOpen: boolean;
	toggleModal: () => void;
	datasetFormats: { id: string; name: string }[];
}) {
	const [stepAtive, setStepActive] = useState(1);

	// form state (useState per field)
	const [sourceMethod, setSourceMethod] = useState<"file" | "hf">("file");
	const [sourceFile, setSourceFile] = useState<File | null>(null);
	const [hfRepoId, setHfRepoId] = useState<string>("");
	const [detectedSample, setDetectedSample] = useState(0);
	const [detectedFormat, setDetectedFormat] = useState<string>("");
	const [detectedSplit, setDetectedSplit] = useState<string>("train");
	const [revision, setRevision] = useState<string>("");

	// kumpulkan jadi objek StepOneValue untuk dipakai sebagai `value`
	const stepOneValue: StepOneValue = {
		sourceMethod,
		sourceFile,
		hfRepoId,
		split: detectedSplit,
		revision,
		detectedFormat,
		detectedSample,
	};

	// child memanggil onChange(patch) → field masing-masing di-update di sini
	const updateStepOne = (patch: Partial<StepOneValue>) => {
		if ("sourceMethod" in patch) {
			setSourceMethod(patch.sourceMethod ?? "file");
		}
		if ("sourceFile" in patch) {
			setSourceFile(patch.sourceFile ?? null);
		}
		if ("hfRepoId" in patch) {
			setHfRepoId(patch.hfRepoId ?? "");
		}
		if ("split" in patch) {
			setDetectedSplit(patch.split ?? "train");
		}
		if ("revision" in patch) {
			setRevision(patch.revision ?? "");
		}
		if ("detectedFormat" in patch) {
			setDetectedFormat(patch.detectedFormat ?? "");
		}
		if ("detectedSample" in patch) {
			setDetectedSample(patch.detectedSample ?? 0);
		}
	};

	return (
		<Dialog open={isOpen}>
			<form>
				<DialogContent
					className="sm:max-w-3xl w-full"
					showCloseButton={false}>
					<DialogHeader>
						<DialogTitle>Upload New Knowledge Dataset</DialogTitle>
						<DialogDescription>
							Enter the details for your new knowledge item.
						</DialogDescription>
					</DialogHeader>
					{stepAtive <= steps.length && (
						<div className="grid grid-cols-5">
							{steps.map((step) => (
								<div
									key={step.id}
									className="flex flex-col justify-center items-center gap-1 p-2 rounded-lg">
									<div
										className={clsx(
											"w-10 h-10 flex items-center justify-center",
											stepAtive === step.id
												? "bg-black text-white rounded-full"
												: stepAtive > step.id
													? "bg-black text-white rounded-full"
													: "bg-gray-200 text-gray-500 rounded-full",
										)}>
										{stepAtive > step.id ? (
											<Check className="w-5 h-5" />
										) : (
											<span className="font-semibold">
												{step.id}
											</span>
										)}
									</div>
									<span className="font-semibold sm:flex hidden">
										{step.name}
									</span>
									<p className="text-sm text-muted-foreground sm:flex hidden">
										{step.description}
									</p>
								</div>
							))}
						</div>
					)}

					<ScrollArea className="group max-h-125">
						{stepAtive === 1 && (
							<DatasetStepOne
								value={stepOneValue}
								onChange={updateStepOne}
							/>
						)}
						{stepAtive === 2 && <DatasetStepTwo />}
						{stepAtive === 3 && <DatasetStepThree />}
						{stepAtive === 4 && <DatasetStepFour />}
						{stepAtive === 5 && <DatasetStepFive />}
						{stepAtive === 6 && <AddDatasetResponse />}
					</ScrollArea>

					<DialogFooter>
						{stepAtive > 1 && (
							<Button
								variant="outline"
								onClick={() => setStepActive(stepAtive - 1)}>
								Back
							</Button>
						)}
						{stepAtive === 1 && (
							<DialogClose
								onClick={toggleModal}
								render={
									<Button variant="outline">Cancel</Button>
								}
							/>
						)}
						{stepAtive <= steps.length && (
							<Button
								type="button"
								onClick={() => setStepActive(stepAtive + 1)}>
								Next
							</Button>
						)}
						{stepAtive === steps.length && (
							<Button type="submit">
								Ingest Version Manifest
							</Button>
						)}
					</DialogFooter>
					<DialogPrimitive.Close
						onClick={toggleModal}
						data-slot="dialog-close"
						render={
							<Button
								variant="ghost"
								className="absolute top-4 right-4 bg-secondary"
								size="icon-sm"
							/>
						}>
						<XIcon />
						<span className="sr-only">Close</span>
					</DialogPrimitive.Close>
				</DialogContent>
			</form>
		</Dialog>
	);
}
