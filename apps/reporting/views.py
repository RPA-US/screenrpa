import base64
import io
import json
import os
import pickle
import re
from tempfile import NamedTemporaryFile
#from tkinter import Image
from django.http import FileResponse, HttpResponse, HttpResponseNotFound, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
import datetime
import docx
from docx import Document
from docx.shared import Inches
from django.core.exceptions import ValidationError
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from django.views.generic import ListView, DetailView, CreateView
import pandas as pd
from sklearn.tree import export_graphviz
from .models import PDD
from apps.analyzer.models import CaseStudy
from django.utils.translation import gettext_lazy as _
from apps.analyzer.models import CaseStudy, Execution # add this
from .forms import ReportingForm
from django.urls import reverse

import os
import zipfile
from django.http import HttpResponse
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _
from django.conf import settings
from django.shortcuts import get_object_or_404

import pydotplus
#import pypandoc
from PIL import Image, ImageDraw
#import subprocess
import aspose.words as aw
##################################
from graphviz import Source, Digraph
from tempfile import NamedTemporaryFile
from apps.processdiscovery.utils import Process

import pygraphviz as pgv 


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
    document.core_properties.title = _("Process Definition Document")
    document.core_properties.comments = _("This is a first version of your RPA Process Definition Document.")
    document.core_properties.category = _("PDDs")
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
    document.add_paragraph(_('Process Definition Document'), custom_style).alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph(_('Version 1.0'), custom_style).alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_page_break()
    
    return document
    
def pdd_table_of_contents(document):
    # add a table of contents
    paragraph = document.add_paragraph()
    run = paragraph.add_run(_('Table of Contents'))
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
    fldChar3.text = _("Right-click to update field.")

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
    document.add_heading(_('1. Introduction'), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('1. Introduction')

    # add body text for section 1
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('This document outlines the process definition for...'))
    document.add_page_break()

    # add section 2: process overview
    document.add_heading('2. Process Overview', level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('2. Process Overview')

    # add body text for section 2
    body_text = document.add_paragraph()
    body_text.style = 'normal'
    body_text.add_run(_('This process consists of the following steps:'))
    # add a numbered list for the steps in section 2
    numbered_list = document.add_paragraph()
    numbered_list.style = 'List Number'
    numbered_list.add_run(_('Step 1: ...'))
    numbered_list.add_run(_('\nStep 2: ...'))
    numbered_list.add_run(_('\nStep 3: ...'))
    
    # add section 3: roles and responsibilities
    document.add_page_break()
    document.add_heading(_(), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('3. Roles and Responsibilities')

    # add body text for section 3
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('The following roles and responsibilities are involved in this process:'))

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
    heading_cells[0].text = _('Role')
    heading_cells[1].text = _('Responsibilities')

    # add the content for the remaining rows in the table
    row_cells = table.rows[1].cells
    row_cells[0].text = _('Manager')
    row_cells[1].text = _('Approves final output of the process')
    row_cells = table.rows[2].cells
    row_cells[0].text = _('Analyst')
    row_cells[1].text = _('Conducts analysis for the process')
    row_cells = table.rows[3].cells
    row_cells[0].text = _('Developer')
    row_cells[1].text = _('Implements changes to the process')

    # add section 4: process flow
    document.add_page_break()
    document.add_heading(_('4. Process Flow'), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('4. Process Flow')

    # add body text for section 4
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('The process flow for this process is as follows:'))
    # add an image for the process flow in section 4
    document.add_picture('process_flow.png', width=Inches(6))
    # add section 5: process metrics
    document.add_page_break()
    document.add_heading(_('5. Process Metrics'), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('5. Process Metrics')

    # add body text for section 5
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('The following metrics are tracked for this process:'))
    # add a bulleted list for the metrics in section 5
    bulleted_list = document.add_paragraph()
    bulleted_list.style = 'List Bullet'
    # add the metrics to the bulleted list
    bulleted_list.add_run(_('Metric 1: ...'))
    bulleted_list.add_run(_('\nMetric 2: ...'))
    bulleted_list.add_run(_('\nMetric 3: ...'))
    
    # add section 6: process improvement
    document.add_page_break()
    document.add_heading(_('6. Process Improvement'), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('6. Process Improvement')

    # add body text for section 6
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('The following improvement opportunities have been identified for this process:'))
    # add a numbered list for the improvement opportunities in section 6

    # Adding list of style name 'List Number'
    document.add_heading('Style: List Number', 3)
    # Adding points to the list named 'List Number'
    document.add_paragraph(_('Opportunity 1: ...'), style='List Number')
    document.add_paragraph(_('Opportunity 2: ...'), style='List Number')
    document.add_paragraph(_('Opportunity 3: ...'), style='List Number')
    
    # Adding list of style name 'List Number 2'
    document.add_heading('Style: List Number 2', 3)
    # Adding points to the list named 'List Number 2'
    document.add_paragraph(_('Opportunity 1: ...'), style='List Number 2')
    document.add_paragraph(_('Opportunity 2: ...'), style='List Number 2')
    document.add_paragraph(_('Opportunity 3: ...'), style='List Number 2')
    
    # Adding list of style name 'List Number 3'
    document.add_heading('Style: List Number 3', 3)
    # Adding points to the list named 'List Number 3'
    document.add_paragraph(_('Opportunity 1: ...'), style='List Number 3')
    document.add_paragraph(_('Opportunity 2: ...'), style='List Number 3')
    document.add_paragraph(_('Opportunity 3: ...'), style='List Number 3')    


    # add section 7: glossary
    document.add_page_break()
    document.add_heading(_(), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('7. Glossary')
    
    # add body text for section 7
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('The following terms are used throughout this document:'))
    
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
    heading_cells[0].text = _('Term')
    heading_cells[1].text = _('Definition')
    
    # add the content for the remaining rows in the table
    row_cells = table.rows[1].cells
    row_cells[0].text = _('Process')
    row_cells[1].text = _('A set of activities that transform inputs into outputs')
    row_cells = table.rows[2].cells
    row_cells[0].text = _('Metric')
    row_cells[1].text = _('A quantifiable measure used to track the performance of a process')

    # save the document
    document.save('Process Definition Document.docx')
    
    return document
    
    
# TODO:
class ReportGenerateView(DetailView):
    def get(self, request, *args, **kwargs):
        print("se ejecuta")
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
            return HttpResponse(_("File does not exist"), status=404)

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

    def get_context_data(self, **kwargs):
        context = super(ReportListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Search if s is a query parameter
        search = self.request.GET.get("s")
        # Filtra los objetos por case_study_id
        if search:
            queryset = PDD.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user, case_study__title__icontains=search).order_by('-created_at')
        else:
            queryset = PDD.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')
        
        return queryset

#############################################################################################
#############################################################################################
#############################################################################################
## VERSIONES DE ALE

# class ReportGenerateView_ALE(DetailView):
#     def get(self, request, *args, **kwargs):
#         model = Execution
#         execution = get_object_or_404(Execution, id=kwargs["execution_id"]) 

#         pdd = PDD.objects.create(
#             execution=execution,
#             user=request.user  
#         )
#         report_path = os.path.join(execution.exp_folder_complete_path,'exec_'+str(execution.id)+'_reports', 'report.docx')

#         # Simulate report creation (you should replace this with your actual report generation logic)
#         with open(report_path, 'wb') as file:
#             file.write(b'PDF content or whatever content you generate')

#         pdd.file.name = report_path
#         pdd.save()

#         return response
    

#########################################################################


def tree_to_png(path_to_tree_file):
    # Load the decision tree data
    try:
        with open('/screenrpa/' + path_to_tree_file, 'rb') as file:
            loaded_data = pickle.load(file)
        loaded_classifier = loaded_data['classifier']
        loaded_feature_names = loaded_data['feature_names']
        loaded_class_names = [str(item) for item in loaded_data['class_names']]
    except FileNotFoundError:
        print(f"File not found: {path_to_tree_file}")
        return None
    
    dot_data = io.StringIO()
    export_graphviz(loaded_classifier, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, feature_names=loaded_feature_names, class_names=loaded_class_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    png_image = graph.create_png()

    # Save the image to a temporary file
    temp_file = NamedTemporaryFile(delete=False, suffix='.png')
    with open(temp_file.name, 'wb') as file:
        file.write(png_image)
    
    return temp_file.name


#########################################
def dot_to_png(dot_path):
    # Cargar el contenido del archivo .dot
    with open(dot_path, 'r') as file:
        dot_content = file.read()
    try:
        graph = Source(dot_content)
        graph.format = 'png'
        
        # Guardar la imagen a un archivo temporal
        temp_file = NamedTemporaryFile(delete=False, suffix='.png')
        graph_path = graph.render(filename=temp_file.name, format='png', cleanup=True)
        
        return graph_path 
    
    except Exception as e:

        print(f"Error al procesar el gráfico: {e}")
        return None

#############################################################################################
class ReportCreateView(CreateView):
    model = PDD
    form_class = ReportingForm
    template_name = "reporting/create.html"

    def get_context_data(self, **kwargs):
        context = super(ReportCreateView, self).get_context_data(**kwargs)
        
        execution_id = self.kwargs.get('execution_id')
        execution = Execution.objects.get(pk=execution_id)
        context['execution'] = execution
        reports = PDD.objects.filter(execution=execution).order_by('-created_at')
        context['reports'] = reports
        return context 
    
    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError(_("User must be authenticated."))
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.execution = Execution.objects.get(pk=self.kwargs.get('execution_id'))
        saved = self.object.save()

        self.report_generate(self.object)

        return HttpResponseRedirect(self.get_success_url())
    
    def report_generate(self, report):
        execution= self.get_context_data()['execution']
        scenarios_to_study= self.get_context_data()['execution'].scenarios_to_study

        for scenario in scenarios_to_study:

            report_directory = os.path.join(execution.exp_folder_complete_path, scenario+"_results")
            report_path = os.path.join(report_directory, f'report_{report.id}.docx')

            # Ensure the directory exists
            os.makedirs(report_directory, exist_ok=True)

            report_define(report_directory, report_path, execution, report, scenario)
            # with open(report_path, 'wb') as file:
            #     file.write(b'PDF content or whatever content you generate')
        
#############################################################################################


def report_define(report_directory, report_path, execution,  report, scenario):
    template_path = "/screenrpa/apps/templates/reporting/report_template.docx"
    doc = Document(template_path)
    colnames=execution.case_study.special_colnames
    ###############3
    paragraph_dict = {}
    for i, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text
        if text.startswith('[') and text.endswith(']'): 
            paragraph_dict[text] = i 

    for key, value in paragraph_dict.items():
        print(f"Texto del párrafo: {key}\nÍndice: {value}\n")

    ############################ INTRODUCION
    
    purpose= doc.paragraphs[paragraph_dict['[PURPOSE]']]
    purpose.text = report.purpose
    #paragraph.style = doc.styles['Normal']

    purpose= doc.paragraphs[paragraph_dict['[OBJECTIVE]']]
    purpose.text = report.objective

    ############################################# AS IS PROCESS DESCRPTION: PROCESS OVERVIEW

    if report.process_overview:
        title= doc.paragraphs[paragraph_dict['[TITLE]']]
        title.text = execution.process_discovery.title

    ############################ AS IS PROCESS DESCRPTION: APPLICATIONS USED
    if report.applications_used:
        nameapps= doc.paragraphs[paragraph_dict['[DIFERENT NAMEAPPS]']]
        applications_used(nameapps, execution, scenario, colnames)
        #nameapps.style = doc.styles['ListBullet'] --> add_paragraph('text', style='ListBullet')
        

    ##########################3 AS IS PROCESS MAP
    if report.as_is_process_map:
        bpmn= doc.paragraphs[paragraph_dict['[.BPMN]']]
        path_to_tree_file = os.path.join(execution.exp_folder_complete_path, scenario+"_results", "bpmn.dot")
        run = bpmn.add_run()
        run.add_picture(dot_to_png(path_to_tree_file), width=Inches(6))
        run.add_break()

    
    #############################3 DETAILS AS IS PROCESS ACTIONS
    
    if report.detailed_as_is_process_actions:
        #meter diagrama de decision tree
        decision_tree= doc.paragraphs[paragraph_dict['[DECISION TREE]']]
        path_to_tree_file = os.path.join(execution.exp_folder_complete_path, scenario+"_results", "decision_tree.pkl")
        run = decision_tree.add_run()
        run.add_picture(tree_to_png(path_to_tree_file), width=Inches(6))
        run.add_break()
        #decision_tree.text = ''
        #detailes_as_is_process_actions(doc, paragraph_dict, execution, scenario)

        detailes_as_is_process_actions(doc, paragraph_dict, scenario, execution, colnames)
     
    
    #############################3 INPUT DATA DESCRPTION
    if report.input_data_description:
        original_log= doc.paragraphs[paragraph_dict['[ORIGINAL LOG]']]
        df_logcsv = pd.read_csv(os.path.join(execution.exp_folder_complete_path, scenario, 'log.csv'))
        input_data_descrption(doc, original_log, execution, scenario, df_logcsv)



    doc.save(report_path)

    convert_docx_to_pdf(report_path, report_path.replace('.docx', '.pdf'))

#################################################################################

def applications_used(nameapps, execution, scenario, colnames):
    logcsv_directory = os.path.join(execution.exp_folder_complete_path,scenario,'log.csv')
    df_logcsv = pd.read_csv(logcsv_directory)
    unique_names = df_logcsv[colnames['NameApp']].unique()

    for name in unique_names:
        run = nameapps.add_run('\n• ' + name)
        run.add_break()
        row = df_logcsv[df_logcsv[colnames['NameApp']] == name].iloc[0]
        screenshot_filename = row[colnames['Screenshot']]
        screenshot_directory = os.path.join(execution.exp_folder_complete_path, scenario, screenshot_filename)
        
        if os.path.exists(screenshot_directory):
        
            nameapps.add_run().add_picture(screenshot_directory, width=Inches(6))
        else:
            print(f"Image not found for {name}: {screenshot_directory}")
#############################################################################################
def input_data_descrption(doc, original_log, execution, scenario, df_logcsv):
        
        table = doc.add_table(rows=(df_logcsv.shape[0] + 1), cols=df_logcsv.shape[1])

        # Insertar los nombres de las columnas
        for j, col in enumerate(df_logcsv.columns):
            table.cell(0, j).text = col

        # Insertar los datos del DataFrame
        for i in range(df_logcsv.shape[0]):
            for j in range(df_logcsv.shape[1]):
                table.cell(i + 1, j).text = str(df_logcsv.iloc[i, j])

        tbl, p = table._tbl, original_log._p
        p.addnext(tbl)
###################33333

def convert_docx_to_pdf(dx_path, pdf_path):
    
    doc = aw.Document(dx_path)
    doc.save(pdf_path)


# def convert_docx_to_pdf2(dx_path, pdf_path):
#     extra_args = ['--pdf-engine-opt', '-dPDFSETTINGS=/prepress']
#     #pypandoc.convert_file(dx_path, 'pdf', outputfile=pdf_path, extra_args=extra_args)
#     pypandoc.convert_file(dx_path, 'pdf', outputfile=pdf_path, extra_args=extra_args)
##################################################33

def detailes_as_is_process_actions(doc, paragraph_dict, scenario, execution, colnames):
    
    def variant_column(df):
        sequence_to_variant = {}
        next_variant_number = 1
        # Función para generar el mapeo de variantes y asignar valores a la columna Variant2
        def assign_variant(trace):
            nonlocal next_variant_number  # Declarar next_variant_number como global
            cadena = ""
            for e in trace['activity_label'].tolist():
                cadena = cadena + str(e)
            if cadena not in sequence_to_variant:
                sequence_to_variant[cadena] = next_variant_number
                next_variant_number += 1
            trace['Variant2'] = sequence_to_variant[cadena]
            return trace

        # Aplicar la función a cada grupo de trace_id y asignar el resultado al DataFrame original
        df=df.groupby('trace_id').apply(assign_variant).reset_index(drop=True)
        return df
    
    
           
        
    def cambiar_color_nodos_y_caminos(labels, path_to_dot_file):
        # Cargar el contenido del archivo .dot
        with open(path_to_dot_file, 'r') as file:
            dot_content = file.read()
        # Crear un grafo desde el contenido del archivo .dot
        grafo = pgv.AGraph(string=dot_content)
        
        # Crear un conjunto de nodos que necesitamos colorear
        nodos_a_colorear = set()
        labels = [str(elemento) for elemento in labels]
        # Recorrer todos los nodos del grafo y colorear los nodos correspondientes
        for nodo in grafo.nodes():
            if nodo.attr['label'] in labels:
                nodo.attr['fillcolor'] = "yellow"
                nodo.attr['style'] = 'filled'
                nodos_a_colorear.add(nodo.name)
            elif nodo.attr['label'] == "":
                nodo.attr['label'] = " "
                nodo.attr['fillcolor'] = nodo.attr['fillcolor']
                nodo.attr['style'] = nodo.attr['style']
            else:
                nodo.attr['style'] = 'filled'
                

        # Recorrer todas las aristas del grafo y colorear las que conectan nodos en la lista
        # Identificar y colorear los nodos de tipo "diamond" entre los nodos etiquetados
        for edge in grafo.edges():
            if edge[0] in nodos_a_colorear and edge[1] not in nodos_a_colorear:
                nodo_destino = grafo.get_node(edge[1])
                if nodo_destino.attr['shape'] == 'diamond':
                    nodo_destino.attr['fillcolor'] = "yellow"
                    nodo_destino.attr['style'] = 'filled'
                    nodos_a_colorear.add(nodo_destino.name)
            elif edge[1] in nodos_a_colorear and edge[0] not in nodos_a_colorear:
                nodo_origen = grafo.get_node(edge[0])
                if nodo_origen.attr['shape'] == 'diamond':
                    nodo_origen.attr['fillcolor'] = "yellow"
                    nodo_origen.attr['style'] = 'filled'
                    nodos_a_colorear.add(nodo_origen.name)

        # Colorear las aristas conectadas entre los nodos coloreados
        for edge in grafo.edges():
            if edge[0] in nodos_a_colorear and edge[1] in nodos_a_colorear:
                edge.attr['color'] = "blue"
                edge.attr['penwidth'] = 2.0

        # Configurar atributos globales del grafo
        grafo.graph_attr.update(bgcolor='white', rankdir='LR')
        grafo.graph_attr['overlap'] = 'false'
        grafo.format = 'png'

        # Guardar la imagen a un archivo temporal
        temp_file = NamedTemporaryFile(delete=False, suffix='.png')
        grafo.draw(temp_file.name, prog='dot', format='png')
        graph_path = temp_file.name

        return graph_path

        

    def lectura_traceability(json_path):
        # Leer el contenido del fichero JSON
        with open(json_path, 'r') as file:
            traceability = json.load(file)
            return traceability
        

    def get_branch_condition(decision_point, branch_number):
        branches = decision_point.get('branches', [])
        rules = decision_point.get('rules', {})
        
        for branch in branches:
            if branch['label'] == str(branch_number):
                res= rules.get(str(branch_number))
                break
            else: 
                res= "No se han encontrado las reglas asociadas a esta rama."
        return res
    
## devuelve un diccionario cuyas claves son las prev act delos punto de decisiones que hay en una varianye y 
# de valor los json del punto de dceision

    def get_decision_points_for_branches(data, branch_ids):
        result = {}

        def search_decision_points(decision_points, branch_ids):
            for dp in decision_points:
                for branch in dp['branches']:
                    if branch['id'] in branch_ids:
                        result[dp['prevAct']] = dp
                    search_decision_points(branch.get('decision_points', []), branch_ids)

        search_decision_points(data['decision_points'], branch_ids)
        return result
    
#extrae una lista con todos los id de todos los puntos de decision de una variante  
    def extract_decision_point_ids(decision_points):
        ids = []
    
        def extract_ids(dp_list):
            for dp in dp_list:
                ids.append(dp['id'])
                for branch in dp['branches']:
                    extract_ids(branch['decision_points'])
        
        extract_ids(decision_points)
        return ids
    #def explicabilidad_decisions(decision_points, variant_decision_points):


    def explicabilidad_actions(decision_tree, activity,group, colnames):

        def pintar_imagen(k, action_dict):
            event_description = None
            image_filename = None

            action = pd.DataFrame(action_dict[k])
            #action= action_dict[k]

            decision_tree.add_run().add_break()
            decision_tree.add_run(f'Acción {k}')
            decision_tree.add_run().add_break()
            
            if action['MorKeyb'].iloc[0] == 1: #colnames['EventType']
                # Calcular la media de Coor_X y Coor_Y
                mean_x = action[colnames['CoorX']].mean()
                mean_y = action[colnames['CoorY']].mean()
                event_description = f"The user clicks at point ({mean_x}, {mean_y})"

                # Cargar la imagen correspondiente
                screenshot_filename = action[colnames['Screenshot']].iloc[0]
                path_to_image = os.path.join(execution.exp_folder_complete_path, scenario, screenshot_filename)
                image_filename = action[colnames['Screenshot']].iloc[0] if 'Screenshot' in action.columns else None
            else:
                    # Mostrar el valor de TextInput si existe, de lo contrario imprimir "No TextInput"
                text_input = action['features.experiment.GUI_category.name.TextInput'].iloc[0] if 'TextInput' in action.columns and not pd.isnull(action['TextInput'].iloc[0]) else "No TextInput"
                event_description = f'The user writes "{text_input}"'
            
            decision_tree.add_run().add_break()
            if event_description:
                decision_tree.add_run(event_description + '\n')
            decision_tree.add_run().add_break()

            if image_filename:
                with Image.open(path_to_image) as img:
                    draw = ImageDraw.Draw(img)
                    # Dibujar un pequeño cuadrado alrededor de las coordenadas medias
                    box_size = 10  # Ajustar el tamaño del cuadrado según sea necesario
                    left = mean_x - box_size / 2
                    top = mean_y - box_size / 2
                    right = mean_x + box_size / 2
                    bottom = mean_y + box_size / 2
                    draw.rectangle([left, top, right, bottom], outline="red", width=2)
                    # Convertir la imagen a un objeto byte para insertar en docx
                    image_stream = io.BytesIO()
                    img.save(image_stream, 'PNG')
                    image_stream.seek(0)
                        
                    decision_tree.add_run().add_picture(image_stream, width=Inches(4))
                decision_tree.add_run().add_break()
        #####################################

        decision_tree.add_run().add_break()
        decision_tree.add_run(f'Activity {activity}\n').bold = True
        decision_tree.add_run().add_break()

        ############################ EXPLICABILIDAD DE LAS ACTIVIDADES Y ACCIONES
        # Filtrar por actividad
        activity_group = group[group['activity_label'] == activity]

        # Agrupar por trace_id dentro de activity_group
        activity_actions_group = activity_group.groupby('trace_id')
        
        # Verificar si todos los grupos tienen una fila
        all_single_action = all(len(actions) == 1 for _, actions in activity_actions_group)
        # Iterar sobre cada grupo y determinar el número de acciones

        

        if all_single_action: # ACTIVIDAD CON UNA ACCION
            action_dict = {}
            for i, (activity_label, action) in enumerate(activity_group.groupby('activity_label'), start=1):
                action_dict[i] = action

            for k in sorted(action_dict.keys()):
                pintar_imagen(k, action_dict)    
                
             
        else: #ACTIVIDADES CON MAS DE UNA ACCION 
            action_dict = {}  # Crear el diccionario vacío    
            for i, (activity_label, action_group) in enumerate(activity_actions_group):
                #crear un diccionario, cuyos claves sean el indice de accion (accion 1, 2, 3) y los valores sea el grupo de filas de la accion
                            
                for j, (index, action) in enumerate(action_group.iterrows(), start=1):
                    #relleno el diccionario
                    
                    if j not in action_dict:
                        action_dict[j] = []
                    # Relleno el diccionario
                    action_dict[j].append(action)
                    #action_dict[j] = action_dict[j].append(action[1])

            #recorrer las claves del diccionario
            for k in sorted(action_dict.keys()):
                pintar_imagen(k, action_dict)
                    

                
            ############################################
          
############################################################3
    def process_variant_group(group, traceability, path_to_dot_file, colnames):
        #prev_act = traceability['decision_points'][0]['prevAct']
        decision_point = traceability['decision_points'][0]
        
        #print(decision_point)
        first_dp_id= decision_point['id']
        #ir acumulando las id de los puntos de decision de cada variante
        columnas_id_ramas=[]
        for pd in extract_decision_point_ids(traceability['decision_points']):
            columnas_id_ramas= columnas_id_ramas + group[pd].unique().tolist()
            variant_decision_points = get_decision_points_for_branches(traceability, columnas_id_ramas)
        #añadir el primer punto de decision al diccionario
        variant_decision_points[decision_point['prevAct']]= decision_point
        
        variant = group['Variant2'].iloc[0]
        decision_tree.add_run().add_break()
        decision_tree.add_run(f'#####Variant {variant}\n').bold = True
        decision_tree.add_run().add_break()

        #actividades
        activities = group['activity_label'].unique().tolist() #en funcion de si las activity label se pueden repetir


        #meto diagrama bpmn resaltado de variantes
        run = decision_tree.add_run()
        run.add_picture(cambiar_color_nodos_y_caminos(activities, path_to_dot_file), width=Inches(6))
        run.add_break()


        for i, activity in enumerate(activities):

            #EXPLICAR ACTIVIDADES Y ACCIONES
            explicabilidad_actions(decision_tree, activity,group, colnames)
            
            ############################

            #EXPLICABILIDAD DE LAS DECISIONES
            if str(activity) in variant_decision_points:
                decision_point = variant_decision_points[str(activity)]
                num_ramas = len(decision_point['branches'])
                next_activity = activities[i+1] if i+1 < len(activities) else None
                condition = get_branch_condition(decision_point, next_activity) if next_activity else "N/A"

                decision_tree.add_run().add_break()      
                decision_tree.add_run(f'En esta actividad hay un "decision point" con {num_ramas} ramas. En el caso de esta variante (VARIANTE {variant}) se va por la rama {next_activity} y se cumple que: {condition} \n')
                decision_tree.add_run().add_break() 
    
###############
    
    decision_tree= doc.paragraphs[paragraph_dict['[DECISION TREE]']]
    #quitar lo del delimiter, esto lo he puesto porque al modificar un csv me lo guardaba con ; en lugar de ,, pero se genrarn con , de normal
    #df = pd.read_csv(os.path.join(execution.exp_folder_complete_path, scenario+'_results', 'pd_log.csv'), delimiter=';')
    #df = pd.read_csv(os.path.join(execution.exp_folder_complete_path, scenario+'_results', 'pd_log-2acciones.csv'), delimiter=';')
    df = pd.read_csv(os.path.join(execution.exp_folder_complete_path, scenario+'_results', 'pd_log.csv'))
    df2= variant_column(df)
    traceability= lectura_traceability(os.path.join(execution.exp_folder_complete_path, scenario+'_results', 'traceability.json'))
    path_to_dot_file = os.path.join(execution.exp_folder_complete_path, scenario+"_results", "bpmn.dot")
    df2.groupby('Variant2').apply(lambda group: process_variant_group(group,traceability,path_to_dot_file, colnames))
    


###################################################################
#####################################################################

def deleteReport(request):
    report_id = request.GET.get("report_id")

    removed_report = PDD.objects.get(id=report_id)
    if request.user.id != removed_report.user.id:
        raise Exception(_("This case study doesn't belong to the authenticated user"))
    
    #if removed_report.executed != 0:
    #    raise Exception(_("This case study cannot be deleted because it has already been excecuted"))

    execution = removed_report.execution

    scenarios_to_study = execution.scenarios_to_study


    for scenario in scenarios_to_study:
        report_directory = os.path.join(execution.exp_folder_complete_path, scenario+"_results")
        report_path = os.path.join(report_directory, f'report_{report_id}.docx')
        if os.path.exists(report_path):
            os.remove(report_path)
        report_path_pdf = os.path.join(report_directory, f'report_{report_id}.pdf')
        if os.path.exists(report_path_pdf):
            os.remove(report_path_pdf)

    removed_report.delete()

    return HttpResponseRedirect(reverse("analyzer:execution_detail", kwargs={"execution_id": removed_report.execution.id}))

#############################################################################################3



class ReportingConfigurationDetail(DetailView):
    def get(self, request, *args, **kwargs):
        report = get_object_or_404(PDD, id=kwargs["report_id"])
        
        form = ReportingForm(read_only=True, instance=report)  
        context = {"form": form,
            "execution": report.execution,
            }
        return render(request, 'reporting/detail.html', context)
    

##################################################3

def download_report_zip(request, report_id):
    report = get_object_or_404(PDD, pk=report_id)
    execution= report.execution

    zip_filename = f"execution_{execution.id}_reports_{report.id}.zip"

    zip_path = os.path.join(settings.MEDIA_ROOT, zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        scenarios_to_study = execution.scenarios_to_study

        for scenario in scenarios_to_study:

            report_directory = os.path.join(execution.exp_folder_complete_path, scenario + "_results")
            report_filename = f'report_{report.id}.docx'
            report_path = os.path.join(report_directory, report_filename)
            
            new_filename = f'scenario_{scenario}_report_{report.id}.docx'

            # Ensure the file exists before adding to ZIP
            if os.path.exists(report_path):
                zipf.write(report_path, arcname=new_filename)

    with open(zip_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename={zip_filename}'

    os.remove(zip_path)  # Clean up the generated zip file after serving it
    return response


##################################################
def preview_pdf(request, report_id):
    
    report = get_object_or_404(PDD, pk=report_id)
    execution = report.execution
    pdf_path = os.path.join('/screenrpa',execution.exp_folder_complete_path, execution.scenarios_to_study[0]+'_results', f'report_{report.id}.pdf')
    #pdf_path = '/screenrpa/media/unzipped/datos_inciiales_j49gvQs_1714120837/executions/exec_41/sc_0_size50_Balanced_results/calendario-academico.pdf'

    return FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')

    # if os.path.exists(pdf_path):
    #     # Renderizar una plantilla que contenga el iframe
    #     return render(request, 'reporting/preview_pdf.html', {'pdf_path': pdf_path})
    # else:
    #     return HttpResponseNotFound("El documento PDF solicitado no fue encontrado.")