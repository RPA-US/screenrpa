import os
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404
import datetime
import docx
from docx import Document
from docx.shared import Inches
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from django.views.generic import ListView, DetailView
from .models import PDD

    


# Create your views here.

def pdd_define_style(document):
    # =======================================================================================================
    # add a custom style for headings
    heading_style = document.styles.add_style('Heading', WD_STYLE_TYPE.PARAGRAPH)
    heading_style.base_style = document.styles['Heading 1']
    heading_style.font.name = 'Calibri Light'
    heading_style.font.size = docx.shared.Pt(18)
    heading_style.font.color.rgb = docx.shared.RGBColor(0x2d, 0x89, 0xff)

    # add a custom style for the table of contents heading
    toc_heading_style = document.styles.add_style('indice', WD_STYLE_TYPE.PARAGRAPH)
    toc_heading_style.base_style = document.styles['Heading']
    toc_heading_style.font.color.rgb = docx.shared.RGBColor(0x2d, 0x89, 0xff)
    toc_heading_style.font.bold = True
    toc_heading_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # add a custom style for section headings
    section_heading_style = document.styles.add_style('Section Heading', WD_STYLE_TYPE.PARAGRAPH)
    section_heading_style.base_style = document.styles['Heading 2']
    section_heading_style.font.name = 'Calibri Light'
    section_heading_style.font.size = docx.shared.Pt(14)
    section_heading_style.font.color.rgb = docx.shared.RGBColor(0x2d, 0x89, 0xff)
    section_heading_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    section_heading_style.paragraph_format.line_spacing = 1.5

    # add a custom style for body text
    body_style = document.styles.add_style('normal', WD_STYLE_TYPE.PARAGRAPH)
    body_style.base_style = document.styles['Normal']
    body_style.font.name = 'Calibri'
    body_style.font.size = docx.shared.Pt(12)
    # =======================================================================================================
    
    # Create a custom document style
    doc_styles = document.styles
    custom_style = doc_styles.add_style('CustomStyle', 1)
    custom_style.font.name = 'Segoe UI'
    custom_style.font.size = 12
    
    return document, custom_style

def pdd_define_properties(document):
    # add document properties
    document.core_properties.author = "Antonio Martinez Rojas"
    document.core_properties.title = "Process Definition Document"
    document.core_properties.comments = "This is a first version of your RPA Process Definition Document."
    document.core_properties.category = "PDDs"
    document.core_properties.created = datetime.datetime.now()
    
    return document
    
def pdd_add_cover(document, custom_style):
    
    # Set the page orientation to portrait
    section = document.sections[0]
    section.orientation = 1
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    
    # Add a cover page
    document.add_picture('cover_image.png', width=Inches(6))
    document.add_paragraph('Process Definition Document', custom_style).alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph('Version 1.0', custom_style).alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_page_break()
    
    return document
    
def pdd_table_of_contents(document):
    # add a table of contents
    paragraph = document.add_paragraph()
    run = paragraph.add_run('Table of Contents')
    paragraph.style = 'indice'
    # paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Inches(0.2)
    paragraph.paragraph_format.space_before = Inches(0.2)
    # Code for making Table of Contents
    paragraph = document.add_paragraph()
    run = paragraph.add_run()

    fldChar = OxmlElement('w:fldChar')  # creates a new element
    fldChar.set(qn('w:fldCharType'), 'begin')  # sets attribute on element

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')  # sets attribute on element
    instrText.text = 'TOC \\o "1-3" \\h \\z \\u'   # change 1-3 depending on heading levels you need

    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')

    fldChar3 = OxmlElement('w:t')
    fldChar3.text = "Right-click to update field."

    fldChar2.append(fldChar3)

    fldChar4 = OxmlElement('w:fldChar')
    fldChar4.set(qn('w:fldCharType'), 'end')

    r_element = run._r
    r_element.append(fldChar)
    r_element.append(instrText)
    r_element.append(fldChar2)
    r_element.append(fldChar4)

    p_element = paragraph._p
    document.add_page_break()
    
    return document

def ui_screen_trace_back_reporting(case_study_id):
    # create a new document
    document = Document()
    
    document = pdd_define_properties(document)
    document, custom_style = pdd_define_style(document)
    document = pdd_add_cover(document, custom_style)
    document = pdd_table_of_contents(document)
    
    # SECTIONS

    # add section 1: introduction
    document.add_heading('1. Introduction', level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('1. Introduction')

    # add body text for section 1
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run('This document outlines the process definition for...')
    document.add_page_break()

    # add section 2: process overview
    document.add_heading('2. Process Overview', level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('2. Process Overview')

    # add body text for section 2
    body_text = document.add_paragraph()
    body_text.style = 'normal'
    body_text.add_run('This process consists of the following steps:')
    # add a numbered list for the steps in section 2
    numbered_list = document.add_paragraph()
    numbered_list.style = 'List Number'
    numbered_list.add_run('Step 1: ...')
    numbered_list.add_run('\nStep 2: ...')
    numbered_list.add_run('\nStep 3: ...')
    
    # add section 3: roles and responsibilities
    document.add_page_break()
    document.add_heading('3. Roles and Responsibilities', level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('3. Roles and Responsibilities')

    # add body text for section 3
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run('The following roles and responsibilities are involved in this process:')

    # add a table for the roles and responsibilities in section 3
    table = document.add_table(rows=4, cols=2)
    table.style = 'Table Grid'
    table.autofit = False

    # set the width of the columns in the table
    column_widths = (Inches(1.5), Inches(4))
    for i, width in enumerate(column_widths):
        table.columns[i].width = width

    # add the header row for the table
    heading_cells = table.rows[0].cells
    heading_cells[0].text = 'Role'
    heading_cells[1].text = 'Responsibilities'

    # add the content for the remaining rows in the table
    row_cells = table.rows[1].cells
    row_cells[0].text = 'Manager'
    row_cells[1].text = 'Approves final output of the process'
    row_cells = table.rows[2].cells
    row_cells[0].text = 'Analyst'
    row_cells[1].text = 'Conducts analysis for the process'
    row_cells = table.rows[3].cells
    row_cells[0].text = 'Developer'
    row_cells[1].text = 'Implements changes to the process'

    # add section 4: process flow
    document.add_page_break()
    document.add_heading('4. Process Flow', level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('4. Process Flow')

    # add body text for section 4
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run('The process flow for this process is as follows:')
    # add an image for the process flow in section 4
    document.add_picture('process_flow.png', width=Inches(6))
    # add section 5: process metrics
    document.add_page_break()
    document.add_heading('5. Process Metrics', level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('5. Process Metrics')

    # add body text for section 5
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run('The following metrics are tracked for this process:')
    # add a bulleted list for the metrics in section 5
    bulleted_list = document.add_paragraph()
    bulleted_list.style = 'List Bullet'
    # add the metrics to the bulleted list
    bulleted_list.add_run('Metric 1: ...')
    bulleted_list.add_run('\nMetric 2: ...')
    bulleted_list.add_run('\nMetric 3: ...')
    
    # add section 6: process improvement
    document.add_page_break()
    document.add_heading('6. Process Improvement', level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('6. Process Improvement')

    # add body text for section 6
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run('The following improvement opportunities have been identified for this process:')
    # add a numbered list for the improvement opportunities in section 6

    # Adding list of style name 'List Number'
    document.add_heading('Style: List Number', 3)
    # Adding points to the list named 'List Number'
    document.add_paragraph('Opportunity 1: ...', style='List Number')
    document.add_paragraph('Opportunity 2: ...', style='List Number')
    document.add_paragraph('Opportunity 3: ...', style='List Number')
    
    # Adding list of style name 'List Number 2'
    document.add_heading('Style: List Number 2', 3)
    # Adding points to the list named 'List Number 2'
    document.add_paragraph('Opportunity 1: ...', style='List Number 2')
    document.add_paragraph('Opportunity 2: ...', style='List Number 2')
    document.add_paragraph('Opportunity 3: ...', style='List Number 2')
    
    # Adding list of style name 'List Number 3'
    document.add_heading('Style: List Number 3', 3)
    # Adding points to the list named 'List Number 3'
    document.add_paragraph('Opportunity 1: ...', style='List Number 3')
    document.add_paragraph('Opportunity 2: ...', style='List Number 3')
    document.add_paragraph('Opportunity 3: ...', style='List Number 3')    


    # add section 7: glossary
    document.add_page_break()
    document.add_heading('7. Glossary', level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('7. Glossary')
    
    # add body text for section 7
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run('The following terms are used throughout this document:')
    
    # add a table for the glossary in section 7
    table = document.add_table(rows=3, cols=2)
    table.style = 'Table Grid'
    table.autofit = False
    # set the width of the columns in the table

    column_widths = (Inches(1.5), Inches(4))

    for i, width in enumerate(column_widths):
            table.columns[i].width = width
            
    # add the header row for the table
    heading_cells = table.rows[0].cells
    heading_cells[0].text = 'Term'
    heading_cells[1].text = 'Definition'
    
    # add the content for the remaining rows in the table
    row_cells = table.rows[1].cells
    row_cells[0].text = 'Process'
    row_cells[1].text = 'A set of activities that transform inputs into outputs'
    row_cells = table.rows[2].cells
    row_cells[0].text = 'Metric'
    row_cells[1].text = 'A quantifiable measure used to track the performance of a process'

    # save the document
    document.save('Process Definition Document.docx')
    
    return document
    
    
# TODO:
class ReportGenerateView(DetailView):
    def get(self, request, *args, **kwargs):
        cs = get_object_or_404(PDD, id=kwargs["case_study_id"], active=True)
        
        document = ui_screen_trace_back_reporting(cs.id)
        
        # check if the file exists
        if not os.path.exists(document.file.path):
            return HttpResponse("File does not exist", status=404)

        # set the file content type and headers
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{document.name}"'
        
        # read the file and write it to the response
        with open(document.file.path, 'rb') as file:
            response.write(file.read())
        
        return response
    
# TODO:
class ReportDownloadView(DetailView):
    def get(self, request, *args, **kwargs):
        pdd = get_object_or_404(PDD, id=kwargs["report_id"], active=True)
        context = {"pdd": pdd}
        
        # check if the file exists
        if not os.path.exists(pdd.pdd.file.path):
            return HttpResponse("File does not exist", status=404)

        # set the file content type and headers
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{pdd.pdd.name}"'
        
        # read the file and write it to the response
        with open(pdd.pdd.file.path, 'rb') as file:
            response.write(file.read())
        
        return response

# TODO:
class ReportListView(ListView):
    model = PDD
    template_name = "reporting/list.html"
    paginate_by = 50

    def get_queryset(self):
        return PDD.objects.filter(user=self.request.user)
