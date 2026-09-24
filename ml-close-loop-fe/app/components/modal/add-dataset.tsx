import { Button } from "../ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "../ui/dialog";

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { useState } from "react";
import clsx from "clsx";
import {
	DatasetStepTwo,
	type StepTwoValueProps,
} from "./dataset-steps/dataset-step-two";
import {
	DatasetStepOne,
	type StepOneValueProps,
} from "./dataset-steps/dataset-step-one";
import {
	DatasetStepThree,
	type StepThreeValueProps,
} from "./dataset-steps/dataset-step-three";
import {
	DatasetStepFour,
	type StepFourValueProps,
} from "./dataset-steps/dataset-step-four";
import {
	DatasetStepFive,
	type StepFiveValueProps,
} from "./dataset-steps/dataset-step-five";
import {
	AddDatasetResponse,
	type AddDatasetResponseProps,
} from "./dataset-steps/add-dataset-response";
import { targetSchemas } from "~/utils";
import { CheckIcon, XIcon } from "@phosphor-icons/react/dist/ssr";

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
}: {
	isOpen: boolean;
	toggleModal: () => void;
}) {
	const [stepAtive, setStepActive] = useState(1);

	// form state (useState field one)
	const [sourceMethod, setSourceMethod] = useState<"file" | "hf">("file");
	const [sourceFile, setSourceFile] = useState<File | null>(null);
	const [hfRepoId, setHfRepoId] = useState<string>("");
	const [detectedSample, setDetectedSample] = useState(0);
	const [detectedFormat, setDetectedFormat] = useState<string>("");
	const [detectedSplit, setDetectedSplit] = useState<string>("train");
	const [revision, setRevision] = useState<string>("");

	// form state (useState field two)
	const [versionMode, setVersionMode] = useState<"add" | "create">("add");
	const [datasetId, setDatasetId] = useState<string>("");
	const [datasetName, setDatasetName] = useState<string>("");
	const [datasetVersion, setDatasetVersion] = useState<string>("v1.0.0");
	const [intakeNotes, setIntakeNotes] = useState<string>("");

	// form state (useState field three)
	const [targetSchemaId, setTargetSchemaId] = useState<string>(
		targetSchemas[0].id,
	);

	// kumpulkan jadi objek StepOneValue untuk dipakai sebagai `value`
	const stepOneValue: StepOneValueProps = {
		sourceMethod,
		sourceFile,
		hfRepoId,
		split: detectedSplit,
		revision,
		detectedFormat,
		detectedSample,
	};

	const stepTwoValue: StepTwoValueProps = {
		versionMode,
		datasetId,
		datasetName,
		datasetVersion,
		intakeNotes,
	};

	const stepThreeValue: StepThreeValueProps = {
		targetSchemaId,
	};

	const stepFourValue: StepFourValueProps = {
		detectedSample,
		status: "pass",
	};

	const stepFiveValue: StepFiveValueProps = {
		datasetId,
		datasetVersion,
		sourceMethod,
		sourceFile,
		hfRepoId,
		targetSchemaId,
		detectedSample,
	};

	const responseValur: AddDatasetResponseProps = {
		datasetId,
		datasetVersion,
		detectedSample,
	};

	const requestData = {
		file: {
			type: sourceMethod,
			file: sourceFile,
			idHF: hfRepoId,
			dsplit: detectedSplit,
			branch: revision,
		},
		datasetMetadata: {
			id: datasetId,
			name: datasetName,
			version: datasetVersion,
			description: intakeNotes,
			operator: "Maulana M.",
		},
		schema: {
			id: targetSchemaId,
		},
	};

	// child memanggil onChange(patch) → field masing-masing di-update di sini
	const updateStepOne = (patch: Partial<StepOneValueProps>) => {
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

	const updateStepTwo = (patch: Partial<StepTwoValueProps>) => {
		if ("versionMode" in patch) {
			setVersionMode(patch.versionMode ?? "add");
		}
		if ("datasetId" in patch) {
			setDatasetId(patch.datasetId ?? "");
		}
		if ("datasetName" in patch) {
			setDatasetName(patch.datasetName ?? "");
		}
		if ("datasetVersion" in patch) {
			setDatasetVersion(patch.datasetVersion ?? "");
		}
		if ("intakeNotes" in patch) {
			setIntakeNotes(patch.intakeNotes ?? "");
		}
	};

	const updateStepThree = (patch: Partial<StepThreeValueProps>) => {
		if ("targetSchemaId" in patch) {
			setTargetSchemaId(patch.targetSchemaId ?? "");
		}
	};

	const resetForm = () => {
		setStepActive(1);
		setSourceMethod("file");
		setSourceFile(null);
		setHfRepoId("");
		setDetectedSample(0);
		setDetectedFormat("");
		setDetectedSplit("train");
		setRevision("");

		setVersionMode("add");
		setDatasetId("");
		setDatasetName("");
		setDatasetVersion("v1.0.0");
		setIntakeNotes("");

		setTargetSchemaId(targetSchemas[0].id);
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
											<CheckIcon className="w-5 h-5" />
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

					<>
						{stepAtive === 1 && (
							<DatasetStepOne
								value={stepOneValue}
								onChange={updateStepOne}
								setStepActive={setStepActive}
								toggleModal={toggleModal}
							/>
						)}
						{stepAtive === 2 && (
							<DatasetStepTwo
								value={stepTwoValue}
								onChange={updateStepTwo}
								setStepActive={setStepActive}
							/>
						)}
						{stepAtive === 3 && (
							<DatasetStepThree
								value={stepThreeValue}
								onChange={updateStepThree}
								setStepActive={setStepActive}
							/>
						)}
						{stepAtive === 4 && (
							<DatasetStepFour
								value={stepFourValue}
								setStepActive={setStepActive}
							/>
						)}
						{stepAtive === 5 && (
							<DatasetStepFive
								value={stepFiveValue}
								setStepActive={setStepActive}
								submitDataset={() => {
									console.log("submit dataset", requestData);
								}}
							/>
						)}
						{stepAtive === 6 && (
							<AddDatasetResponse
								value={responseValur}
								toggleModal={toggleModal}
								resetForm={resetForm}
							/>
						)}
					</>

					<DialogPrimitive.Close
						onClick={() => {
							toggleModal();
							resetForm();
						}}
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
