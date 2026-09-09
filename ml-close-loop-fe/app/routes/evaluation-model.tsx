import { Package } from "lucide-react";
import { useEffect, useState } from "react";
import { useParams } from "react-router";
import Header from "~/components/ui/header";
import type { TrainingRun } from "~/type";
import { trainingRunsData } from "~/utils";

export default function EvaluationModel() {
    const [model, setModel] = useState<TrainingRun>(trainingRunsData[0]);
    const params = useParams();
    const slug = params.id;
    const [evalOpen, setEvalOpen] = useState(false);

    useEffect(() => {
        setModel(trainingRunsData.find((item) => item.runId === slug) || trainingRunsData[0]);
    }, [slug])

    return (
        <>
            <Header title="Evaluation Model" />
            <div className="pl-4 pr-4 pb-4 h-full max-w-380 w-full mx-auto">
                {model && (
                    <div className="flex flex-col gap-8">
                        <div className="flex gap-4 items-center p-4 border rounded-lg bg-gray-50">
                            <div>
                                <Package className="w-6 h-6 text-gray-500" />
                            </div>
                            <div className="flex flex-col gap-1">
                                <h3>
                                    Evaluation Center & 3 Core Signals
                                </h3>
                                <p>Model: <span>{model.datasetVersion}</span></p>
                            </div>
                        </div>

                        <div>
                            <div>
                                <h3>Mandatory Evaluation Triad</h3>
                                <p>Mandatory Evaluation Triad</p>
                            </div>

                            
                        </div>
                    </div>
                )}
            </div>
        </>
    )
}