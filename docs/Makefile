TITLE=cs244-final-report

PDF=$(TITLE).pdf

$(PDF): src/*.typ
	typst compile src/main.typ $(PDF)
	open $(PDF)

fonts:
	typst fonts

clean:
	rm *.pdf

.PHONY: fonts, clean
