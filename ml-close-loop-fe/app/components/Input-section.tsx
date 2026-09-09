import { Field, FieldLabel } from "./ui/field";
import { Input } from "./ui/input";

export default function InputSection({id, label, onChange, required = false}: {id: string, label: string, onChange: (e: React.ChangeEvent<HTMLInputElement>) => void, required: boolean}) {
    return (
        <Field>
            <FieldLabel>
                {label} {required && (<span className="text-red-500">*</span>)}
            </FieldLabel>
            <Input id={id} onChange={onChange} required={required} />
        </Field>
    )
}